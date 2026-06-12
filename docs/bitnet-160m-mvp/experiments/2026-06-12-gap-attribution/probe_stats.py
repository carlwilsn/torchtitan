#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-layer quantization probe for the gap-attribution arms.

Given a registry config name (any of the 5 arms), builds the live model and
reports, for every probed linear layer (BitLinear in the BitNet arms, the
converter-eligible stock ``Linear`` layers in the stock arm):

- **ternary code fractions** -- fraction of latent weights that absmean
  ternarization maps to -1 / 0 / +1. For arms where weight quant is OFF
  (stock, ``fp16_weights``, ``structure_only``) these are the *hypothetical*
  codes of the same latent weights, so the curves stay comparable across arms;
  the report records whether quantization is actually active in the forward.
- **gamma** -- the absmean weight scale ``mean(|W|)`` (the dequant scale when
  weight quant is on; the same statistic of the FP weight otherwise).
- **latent weight norm** -- Frobenius norm of the full-precision weight.
- **activation absmax stats** -- per-token absmax (the int8 act-quant scale
  basis) of the activations entering each layer on a sample batch: mean /
  median / p99 / max over tokens. Reported both for the raw module input and
  (when the layer has a SubLN pre-norm) for the post-pre-norm activations the
  act-quant actually sees. For stock layers only the raw input is reported.

Usage (from the repo root):

    python docs/bitnet-160m-mvp/experiments/2026-06-12-gap-attribution/probe_stats.py \
        --config llama3_160m_bitnet_fp16_weights \
        [--checkpoint outputs/llama3_160m_bitnet_fp16_weights/checkpoint/step-1700] \
        [--batch-size 2] [--seq-len 128] [--out report.json]

With ``--checkpoint`` the model weights are loaded from a torchtitan DCP
checkpoint (model-only keys) before probing; without it the probe runs on the
seed-locked init (still useful: all BitNet arms are init-identical). The
sample batch is random tokens from the model's vocab with a fixed seed, so the
activation stats are comparable across arms.
"""

import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, REPO_ROOT)

from torchtitan.components.quantization.bitnet import BitLinear  # noqa: E402
from torchtitan.models.common.nn_modules import Linear  # noqa: E402
from torchtitan.models.llama3 import config_registry  # noqa: E402

ARMS = (
    "llama3_160m",
    "llama3_160m_bitnet_structure_only",
    "llama3_160m_bitnet_fp16_weights",
    "llama3_160m_bitnet_no_actquant",
    "llama3_160m_bitnet",
)

# Same exclusion the converter uses: lm_head/output stays stock in all arms,
# so it is excluded from probing in all arms for comparability.
FILTER_FQNS = ("output", "lm_head")


def _tensor_stats(t: torch.Tensor) -> dict:
    t = t.detach().float().flatten()
    return {
        "mean": float(t.mean()),
        "median": float(t.median()),
        "p99": float(torch.quantile(t, 0.99)) if t.numel() > 1 else float(t[0]),
        "max": float(t.max()),
    }


def weight_stats(module: torch.nn.Module, eps: float = 1e-5) -> dict:
    """Ternary code fractions + gamma + latent norm for one layer's weight."""
    w = module.weight.detach().float()
    gamma = w.abs().mean().clamp_min(eps)
    codes = torch.round(w / gamma).clamp(-1, 1)
    n = codes.numel()
    return {
        "gamma_absmean": float(gamma),
        "latent_weight_fro_norm": float(w.norm()),
        "latent_weight_absmax": float(w.abs().max()),
        "ternary_fraction_neg1": float((codes == -1).sum()) / n,
        "ternary_fraction_zero": float((codes == 0).sum()) / n,
        "ternary_fraction_pos1": float((codes == 1).sum()) / n,
        "numel": n,
    }


def build_model(config_name: str, device: torch.device):
    if not hasattr(config_registry, config_name):
        raise SystemExit(
            f"unknown config {config_name!r}; expected one of {ARMS} "
            f"(any registry function in llama3/config_registry.py works)"
        )
    trainer_config = getattr(config_registry, config_name)()
    spec = trainer_config.model_spec
    with torch.device(device):
        model = spec.model.build()
    model.init_states(buffer_device=device)
    model.eval()
    return model


def load_dcp_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> dict:
    """Load a torchtitan DCP checkpoint (model-only flattened keys) in-place."""
    import torch.distributed.checkpoint as dcp

    state_dict = model.state_dict()
    dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "checkpoint": checkpoint_path,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def probed_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """The converter-eligible linears: BitLinear if present, else stock Linear."""
    bit = [
        (fqn, m) for fqn, m in model.named_modules() if isinstance(m, BitLinear)
    ]
    if bit:
        return bit
    return [
        (fqn, m)
        for fqn, m in model.named_modules()
        if isinstance(m, Linear) and not any(f in fqn for f in FILTER_FQNS)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="registry config name (arm)")
    parser.add_argument("--checkpoint", default=None, help="DCP checkpoint dir (step-N)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234, help="sample-batch seed")
    parser.add_argument("--out", default=None, help="write JSON report here")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42)  # match the seed-locked init of the training configs
    model = build_model(args.config, device)

    report: dict = {
        "config": args.config,
        "device": str(device),
        "torch": torch.__version__,
        "weights_from": "init (seed 42)",
    }
    if args.checkpoint:
        report["checkpoint_load"] = load_dcp_checkpoint(model, args.checkpoint)
        report["weights_from"] = args.checkpoint

    modules = probed_modules(model)
    if not modules:
        raise SystemExit("no probed modules found -- is this a llama3 spec?")
    is_bitnet = isinstance(modules[0][1], BitLinear)
    report["probed_module_type"] = type(modules[0][1]).__name__
    report["n_probed_modules"] = len(modules)

    # ---- activation capture: forward-pre hooks on every probed module ----
    captured: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach()

        return hook

    handles = [m.register_forward_pre_hook(make_hook(fqn)) for fqn, m in modules]

    vocab = int(model.tok_embeddings.weight.shape[0])
    torch.manual_seed(args.seed)
    tokens = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device)
    with torch.no_grad():
        model(tokens)
    for h in handles:
        h.remove()

    # ---- per-layer report ----
    layers = {}
    for fqn, m in modules:
        entry = {"weights": weight_stats(m)}
        if is_bitnet:
            entry["activation_quant_active"] = bool(m.activation_quant_enabled)
            entry["weight_quant_active"] = bool(m.weight_quant_enabled)
            entry["has_pre_norm"] = m.pre_norm is not None
        else:
            entry["activation_quant_active"] = False
            entry["weight_quant_active"] = False
            entry["has_pre_norm"] = False

        x = captured.get(fqn)
        if x is not None:
            raw_absmax = x.float().abs().amax(dim=-1).flatten()
            entry["activation_absmax_raw_input"] = _tensor_stats(raw_absmax)
            if is_bitnet and m.pre_norm is not None:
                with torch.no_grad():
                    xn = m.pre_norm(x)
                entry["activation_absmax_post_prenorm"] = _tensor_stats(
                    xn.float().abs().amax(dim=-1).flatten()
                )
        layers[fqn] = entry
    report["layers"] = layers

    # ---- aggregates (medians across layers, for quick cross-arm comparison) ----
    def med(key_fn):
        vals = sorted(key_fn(e) for e in layers.values())
        return vals[len(vals) // 2]

    report["aggregate_median_across_layers"] = {
        "gamma_absmean": med(lambda e: e["weights"]["gamma_absmean"]),
        "latent_weight_fro_norm": med(lambda e: e["weights"]["latent_weight_fro_norm"]),
        "ternary_fraction_zero": med(lambda e: e["weights"]["ternary_fraction_zero"]),
        "activation_absmax_raw_mean": med(
            lambda e: e["activation_absmax_raw_input"]["mean"]
        ),
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"report written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
