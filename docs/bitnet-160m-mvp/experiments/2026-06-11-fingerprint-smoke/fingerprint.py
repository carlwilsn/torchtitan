#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BitNet fingerprint probe — explicit evidence that the BitNet path is real.

Run from the repo root:

    python docs/bitnet-160m-mvp/experiments/2026-06-11-fingerprint-smoke/fingerprint.py

Produces a JSON report (stdout + ``fingerprint_report.json`` next to this
file) answering, with module-level evidence rather than a converter log line:

1. **Conversion fingerprint** — how many Linear *configs* the
   ``BitLinearConverter`` rewrites for ``llama3_160m_bitnet`` vs how many
   ``BitLinear`` *modules* the built model actually contains, with full FQN
   lists. This reconciles the long-standing ``Swapped 84`` vs
   ``bitlinear_count 98`` mismatch: ``QKVLinear`` builds ``wk`` and ``wv``
   from the SAME ``wkv`` config object, so each layer has 6 configs but 7
   modules (84 + 14 = 98).
2. **Ternary weight fingerprint** — for a sample BitLinear weight after real
   init, the quantized forward weight takes exactly the values
   {-gamma, 0, +gamma} (absmean scaling), with the value histogram recorded.
3. **Activation-quant fingerprint** — per-token absmax int8: quantized
   activations land on <= 255 distinct levels per token row.
4. **STE gradient fingerprint** — one forward/backward of the full 160M
   BitNet model on random tokens: loss is finite, latent full-precision
   weights receive nonzero grads, INCLUDING weights whose forward value was
   clipped to a ternary endpoint (|w/gamma| > 1.5 rounds past +-1) — the
   defining property of the identity STE.
5. **Stock-vs-BitNet output fingerprint** — same seed, same tokens: stock and
   BitNet logits differ (quantization is actually in the forward path).
"""

import json
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from torchtitan.components.quantization.bitnet import (  # noqa: E402
    activation_quant,
    BitLinear,
    BitLinearConverter,
    weight_quant,
)
from torchtitan.models.common.nn_modules import Linear  # noqa: E402
from torchtitan.models.llama3 import model_registry  # noqa: E402


def build_model(spec, device: torch.device):
    with torch.device(device):
        model = spec.model.build()
    model.init_states(buffer_device=device)
    return model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict = {"device": str(device), "torch": torch.__version__}

    # ---- 1. Conversion fingerprint: configs converted vs modules built ----
    converter_cfg = BitLinearConverter.Config(
        filter_fqns=["output", "lm_head"],
        activation_quant=True,
        weight_quant=True,
        pre_norm=True,
    )
    # Count converted *configs* by traversing the same way the converter does.
    spec_bitnet = model_registry("160M")  # converter applied manually below
    converted_fqns = []
    converter = BitLinearConverter(converter_cfg)
    # Replicate converter.convert() with FQN capture.
    for fqn, linear_config, parent, attr in spec_bitnet.model.traverse(Linear.Config):
        if isinstance(linear_config, BitLinear.Config):
            continue
        if not converter.filter_fn(linear_config, fqn):
            continue
        converted_fqns.append(fqn)
    # Now actually convert (fresh spec so the capture above didn't mutate).
    spec_bitnet = model_registry("160M", converters=[converter_cfg])

    torch.manual_seed(42)
    model_bitnet = build_model(spec_bitnet, device)

    bitlinear_module_fqns = [
        name for name, m in model_bitnet.named_modules() if isinstance(m, BitLinear)
    ]
    report["conversion"] = {
        "configs_converted": len(converted_fqns),
        "bitlinear_modules_built": len(bitlinear_module_fqns),
        "explanation": "QKVLinear builds wk and wv from the same wkv config object,"
        " so each transformer layer contributes 6 converted configs but 7"
        " BitLinear modules (wq, wk, wv, wo, w1, w2, w3).",
        "converted_config_fqns_sample": converted_fqns[:8],
        "bitlinear_module_fqns_sample": bitlinear_module_fqns[:8],
        "lm_head_type": type(model_bitnet.lm_head).__name__,
        "tok_embeddings_type": type(model_bitnet.tok_embeddings).__name__,
        "layers": len(model_bitnet.layers),
        "param_count": sum(p.numel() for p in model_bitnet.parameters()),
    }
    # Per-layer module count sanity: expect 7 BitLinear per layer.
    per_layer = {}
    for fqn in bitlinear_module_fqns:
        if fqn.startswith("layers."):
            layer = fqn.split(".")[1]
            per_layer[layer] = per_layer.get(layer, 0) + 1
    report["conversion"]["bitlinear_per_layer"] = sorted(set(per_layer.values()))

    # ---- 2. Ternary weight fingerprint ----
    sample_fqn = bitlinear_module_fqns[0]
    sample = dict(model_bitnet.named_modules())[sample_fqn]
    w = sample.weight.detach().float()
    qw = weight_quant(w)
    gamma = w.abs().mean().clamp_min(1e-5).item()
    uniq = torch.unique(qw)
    hist = {f"{v:+.6f}": int((qw == v).sum()) for v in uniq.tolist()}
    report["weight_quant"] = {
        "sample_module": sample_fqn,
        "gamma_absmean": gamma,
        "unique_forward_values": [round(v, 6) for v in uniq.tolist()],
        "n_unique": int(uniq.numel()),
        "is_ternary": int(uniq.numel()) <= 3,
        "histogram": hist,
        "zero_fraction": float((qw == 0).float().mean()),
    }

    # ---- 3. Activation quant fingerprint ----
    torch.manual_seed(0)
    x = torch.randn(4, 64, device=device)
    qx = activation_quant(x)
    levels_per_row = [int(torch.unique(qx[i]).numel()) for i in range(qx.shape[0])]
    report["activation_quant"] = {
        "input_shape": list(x.shape),
        "unique_levels_per_token_row": levels_per_row,
        "max_levels_allowed": 256,
        "all_rows_within_int8_levels": all(n <= 256 for n in levels_per_row),
    }

    # ---- 4. STE gradient fingerprint (full model fwd/bwd) ----
    torch.manual_seed(1234)
    tokens = torch.randint(0, 2048, (2, 128), device=device)
    out = model_bitnet(tokens)
    loss = torch.nn.functional.cross_entropy(
        out.reshape(-1, out.shape[-1]).float(), tokens.reshape(-1)
    )
    loss.backward()
    wgrad = sample.weight.grad
    gamma_t = sample.weight.detach().abs().mean().clamp_min(1e-5)
    clipped_mask = (sample.weight.detach() / gamma_t).abs() > 1.5
    report["ste_gradient"] = {
        "loss": float(loss),
        "loss_finite": bool(torch.isfinite(loss)),
        "sample_weight_grad_nonzero_fraction": float((wgrad != 0).float().mean()),
        "n_forward_clipped_weights": int(clipped_mask.sum()),
        "clipped_weights_grad_nonzero_fraction": (
            float((wgrad[clipped_mask] != 0).float().mean())
            if int(clipped_mask.sum()) > 0
            else None
        ),
        "note": "clipped weights still receiving grad == identity STE active",
    }

    # ---- 5. Stock-vs-BitNet output fingerprint ----
    torch.manual_seed(42)
    model_stock = build_model(model_registry("160M"), device)
    with torch.no_grad():
        out_stock = model_stock(tokens)
        out_bit = model_bitnet(tokens)
    diff = (out_stock - out_bit).abs()
    report["stock_vs_bitnet_forward"] = {
        "same_seed_same_tokens": True,
        "mean_abs_logit_diff": float(diff.mean()),
        "max_abs_logit_diff": float(diff.max()),
        "outputs_differ": bool(diff.max() > 1e-3),
        "stock_bitlinear_modules": sum(
            isinstance(m, BitLinear) for m in model_stock.modules()
        ),
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fingerprint_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nreport written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
