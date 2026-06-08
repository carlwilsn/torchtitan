# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.components.quantization.bitnet import (
    activation_quant,
    BitLinear,
    BitLinearConverter,
    weight_quant,
)
from torchtitan.config import Configurable
from torchtitan.models.common.nn_modules import Linear


def test_weight_quant_forward_values_are_scaled_ternary() -> None:
    w = torch.tensor([[-2.0, -0.2, 0.2, 2.0]], requires_grad=True)
    q = weight_quant(w)
    gamma = w.detach().abs().mean()
    allowed = torch.tensor([-gamma, 0.0, gamma], dtype=q.dtype)
    assert all(torch.any(torch.isclose(v, allowed)) for v in q.flatten())

    q.sum().backward()
    assert w.grad is not None
    assert torch.all(w.grad != 0)


def test_activation_quant_preserves_shape_dtype_and_grad() -> None:
    x = torch.randn(2, 3, 8, dtype=torch.float32, requires_grad=True)
    y = activation_quant(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype

    y.sum().backward()
    assert x.grad is not None
    assert torch.all(x.grad != 0)


def test_bitlinear_forward_backward_with_prenorm() -> None:
    layer = BitLinear(
        BitLinear.Config(
            in_features=8,
            out_features=4,
            bias=False,
            pre_norm=True,
        )
    )
    layer.init_states()

    x = torch.randn(2, 3, 8, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 3, 4)

    y.sum().backward()
    assert layer.weight.grad is not None
    assert layer.pre_norm is not None
    assert layer.pre_norm.weight.grad is not None


class TinyConfigTree(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        keep: Linear.Config
        swap: Linear.Config
        nested: list[Linear.Config]

    def __init__(self, config: Config):
        pass


def test_bitlinear_converter_replaces_matching_linear_configs() -> None:
    tree = TinyConfigTree.Config(
        keep=Linear.Config(in_features=8, out_features=8),
        swap=Linear.Config(in_features=8, out_features=8),
        nested=[Linear.Config(in_features=8, out_features=8)],
    )

    BitLinearConverter(
        BitLinearConverter.Config(filter_fqns=["keep"])
    ).convert(tree)

    assert isinstance(tree.keep, Linear.Config)
    assert not isinstance(tree.keep, BitLinear.Config)
    assert isinstance(tree.swap, BitLinear.Config)
    assert isinstance(tree.nested[0], BitLinear.Config)
