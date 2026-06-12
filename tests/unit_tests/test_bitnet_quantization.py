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


def test_bitlinear_ablation_fp16_weights_skips_weight_quant() -> None:
    """weight_quant=False: forward must use the latent FP weights verbatim."""

    cfg = BitLinear.Config(
        in_features=8, out_features=4, bias=False,
        pre_norm=False, activation_quant=False, weight_quant=False,
    )
    layer = BitLinear(cfg)
    layer.init_states()

    x = torch.randn(2, 8)
    expected = torch.nn.functional.linear(x, layer.weight)
    assert torch.equal(layer(x), expected)


def test_bitlinear_ablation_no_actquant_skips_activation_quant() -> None:
    """activation_quant=False: only weight ternarization remains."""

    from torchtitan.components.quantization.bitnet import weight_quant as wq

    cfg = BitLinear.Config(
        in_features=8, out_features=4, bias=False,
        pre_norm=False, activation_quant=False, weight_quant=True,
    )
    layer = BitLinear(cfg)
    layer.init_states()

    x = torch.randn(2, 8)
    expected = torch.nn.functional.linear(x, wq(layer.weight))
    assert torch.equal(layer(x), expected)


def test_gap_attribution_configs_set_expected_converter_flags() -> None:
    """The three ablation registry configs differ from llama3_160m_bitnet only
    in the converter's quant flags (same seed, steps, lr, filter_fqns)."""

    import pytest

    try:
        from torchtitan.models.llama3.config_registry import (
            llama3_160m_bitnet,
            llama3_160m_bitnet_fp16_weights,
            llama3_160m_bitnet_no_actquant,
            llama3_160m_bitnet_structure_only,
        )
    except ImportError as e:
        # The llama3 package import chain needs a recent torch nightly
        # (e.g. DataParallelMeshDims). Skip on older torch instead of failing.
        pytest.skip(f"llama3 config_registry unavailable in this env: {e}")

    base = llama3_160m_bitnet()
    expectations = {
        llama3_160m_bitnet_fp16_weights: (True, False),
        llama3_160m_bitnet_no_actquant: (False, True),
        llama3_160m_bitnet_structure_only: (False, False),
    }
    def bitlinear_entries(cfg):
        return [
            (fqn, lc)
            for fqn, lc, _parent, _attr in cfg.model_spec.model.traverse(Linear.Config)
            if isinstance(lc, BitLinear.Config)
        ]

    base_entries = bitlinear_entries(base)
    assert len(base_entries) > 0

    for fn, (act_q, w_q) in expectations.items():
        cfg = fn()
        entries = bitlinear_entries(cfg)
        # Same set of converted layers as the MVP BitNet config...
        assert [fqn for fqn, _ in entries] == [fqn for fqn, _ in base_entries], (
            fn.__name__
        )
        # ...with only the quant flags differing.
        for fqn, lc in entries:
            assert lc.activation_quant is act_q, (fn.__name__, fqn)
            assert lc.weight_quant is w_q, (fn.__name__, fqn)
            assert lc.pre_norm is True, (fn.__name__, fqn)
            assert "output" not in fqn and "lm_head" not in fqn, (fn.__name__, fqn)
        # Everything outside the converter must match the MVP BitNet config.
        assert cfg.debug.seed == base.debug.seed == 42
        assert cfg.training.steps == base.training.steps
        assert cfg.optimizer.lr == base.optimizer.lr
        assert cfg.training.seq_len == base.training.seq_len


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
