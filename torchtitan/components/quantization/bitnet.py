# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BitNet b1.58-style quantized Linear layers for TorchTitan.

This module intentionally mirrors the shape of ``float8.py``:

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
from torch import nn

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.tools.logging import logger
from torchtitan.components.quantization.float8 import module_filter_fn


_EPS = 1e-5


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass, identity in the backward pass."""

    return x + (torch.round(x) - x).detach()


def activation_quant(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Per-token absmax int8 activation quantization with STE.

    BitNet b1.58 uses 8-bit activations. For an activation tensor shaped like
    ``[..., hidden]``, we compute a scale per token/vector over the last
    dimension and quantize to ``[-128, 127]``.

    The returned tensor is dequantized back to the input dtype so the rest of
    the model can continue using ordinary PyTorch matmuls during training.
    """

    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    y = (_ste_round(x * scale).clamp(-128, 127)) / scale
    return y.to(dtype=x.dtype)


def weight_quant(w: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Absmean ternary weight quantization with STE.

    Forward values are in ``{-1, 0, 1}`` after scaling by the mean absolute
    latent-weight magnitude. Gradients flow to the latent full-precision
    weights through the STE identity path.
    """

    scale = 1.0 / w.abs().mean().clamp_min(eps)
    u = _ste_round(w * scale).clamp(-1, 1) / scale
    return u.to(dtype=w.dtype)


class BitLinear(ModuleNotFoundError):
    pass
