# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BitNet b1.58-style quantized Linear layers for TorchTitan.

This module intentionally mirrors the extension shape of ``float8.py``:

- ``BitLinear`` is a configurable drop-in replacement for TorchTitan's
  ``models.common.nn_modules.Linear``.
- ``BitLinearConverter`` rewrites ``Linear.Config`` entries in a model config
  before the model is built.

The training implementation keeps full-precision latent weights and uses a
straight-through estimator (STE) through ternary weights. This is the expected
training behavior for BitNet-style layers; the packed 1.58-bit representation is
an inference/deployment concern, not the optimizer state used during training.
"""

from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn.functional as F

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.components.quantization.utils import module_filter_fn
from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


_EPS = 1e-5


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass, identity in the backward pass."""

    return x + (torch.round(x) - x).detach()


def activation_quant(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Per-token absmax int8 activation quantization with STE.

    For an activation tensor shaped like ``[..., hidden]``, compute one scale
    per token/vector over the last dimension, quantize to signed int8 range,
    then dequantize back to the original dtype for ordinary PyTorch matmul.
    """

    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    y = _ste_round(x * scale).clamp(-128, 127) / scale
    return y.to(dtype=x.dtype)


def weight_quant(w: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Absmean ternary weight quantization with STE.

    Forward values are scaled ternary values. Gradients flow to the latent
    full-precision weights through the STE identity path.
    """

    scale = 1.0 / w.abs().mean().clamp_min(eps)
    u = _ste_round(w * scale).clamp(-1, 1) / scale
    return u.to(dtype=w.dtype)


class BitLinear(Module):
    """TorchTitan-compatible BitNet b1.58 linear layer.

    The layer owns the same trainable parameters as ``nn.Linear``: a latent
    full-precision ``weight`` and optional ``bias``. Forward pass uses:

    1. optional RMSNorm/SubLN on activations,
    2. int8 activation quantization (dequantized for training matmul),
    3. ternary absmean weight quantization (dequantized for training matmul),
    4. ordinary ``F.linear``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        activation_quant: bool = True
        weight_quant: bool = True
        pre_norm: bool = True
        eps: float = _EPS

    def __init__(self, config: Config):
        super().__init__()
        self.in_features = config.in_features
        self.out_features = config.out_features
        self.activation_quant_enabled = config.activation_quant
        self.weight_quant_enabled = config.weight_quant
        self.eps = config.eps

        self.weight = torch.nn.Parameter(
            torch.empty(config.out_features, config.in_features)
        )
        if config.bias:
            self.bias = torch.nn.Parameter(torch.empty(config.out_features))
        else:
            self.register_parameter("bias", None)

        self.pre_norm = (
            RMSNorm(
                RMSNorm.Config(
                    normalized_shape=config.in_features,
                    eps=config.eps,
                    elementwise_affine=True,
                )
            )
            if config.pre_norm
            else None
        )

    def reset_parameters(self) -> None:
        # Matches nn.Linear's default fallback when TorchTitan did not provide
        # param_init. In normal Llama configs, Module.init_states uses the
        # copied param_init dict instead.
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        if self.activation_quant_enabled:
            x = activation_quant(x, eps=self.eps)

        w = self.weight
        if self.weight_quant_enabled:
            w = weight_quant(w, eps=self.eps)

        return F.linear(x, w, self.bias)


class BitLinearConverter(QuantizationConverter):
    """Replace matching ``Linear.Config`` entries with ``BitLinear.Config``."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        filter_fqns: list[str] = field(default_factory=lambda: ["output", "lm_head"])
        """Fully qualified name substrings to skip during conversion."""

        activation_quant: bool = True
        weight_quant: bool = True
        pre_norm: bool = True
        eps: float = _EPS
        require_dim_multiple_of_16: bool = False
        """If true, reuse the float8-style multiple-of-16 filter."""

    def __init__(self, config: Config):
        self.config = config
        if config.require_dim_multiple_of_16:
            self.filter_fn = partial(module_filter_fn, filter_fqns=config.filter_fqns)
        else:
            self.filter_fn = lambda _linear_config, fqn: not any(
                filter_fqn in fqn for filter_fqn in config.filter_fqns
            )

    def convert(self, model_config) -> None:
        converted = 0
        for fqn, linear_config, parent, attr in model_config.traverse(Linear.Config):
            # Avoid re-converting configs that are already BitLinear configs.
            if isinstance(linear_config, BitLinear.Config):
                continue
            if not self.filter_fn(linear_config, fqn):
                continue

            new_config = BitLinear.Config(
                in_features=linear_config.in_features,
                out_features=linear_config.out_features,
                bias=linear_config.bias,
                param_init=linear_config.param_init,
                sharding_config=linear_config.sharding_config,
                activation_quant=self.config.activation_quant,
                weight_quant=self.config.weight_quant,
                pre_norm=self.config.pre_norm,
                eps=self.config.eps,
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)
            converted += 1

        logger.info(f"Swapped {converted} Linear layers to BitLinear")


__all__ = [
    "BitLinear",
    "BitLinearConverter",
    "activation_quant",
    "weight_quant",
]
