# Gap attribution — which BitNet component costs the +0.13 nats? (2026-06-12)

## Question

The 1700-step seed-locked MVP run ([`../../results/`](../../results/)) showed
BitNet 160M trailing stock by **+0.131 train / +0.123 val nats** at step 1700.
This experiment attributes that gap to a component instead of re-measuring it.

BitLinear's forward differs from stock `Linear` in exactly three ways:

1. **Structure** — an extra learnable pre-RMSNorm (SubLN-style) in front of every
   converted linear;
2. **Activation quantization** — per-token absmax int8 quant/dequant with STE;
3. **Weight ternarization** — absmean ternary quant/dequant with STE.

The lm_head/output projection is stock in ALL configs (`filter_fqns=["output",
"lm_head"]`), so a separate "FP16 head" ablation is redundant — that is already
the baseline behavior of `llama3_160m_bitnet`.

## Variants

All are registry configs in `torchtitan/models/llama3/config_registry.py`. Each
inherits `llama3_160m` (seed 42, same shape, same LR/schedule/steps) and applies
`BitLinearConverter` with `pre_norm=True` and the same `filter_fqns` — only the
two quant flags differ. The stock path is untouched: ablations are additive
config functions, and `llama3_160m` / `llama3_160m_bitnet` are byte-identical to
the MVP run.

| Config name | act quant | weight ternary | What it isolates |
| --- | :-: | :-: | --- |
| `llama3_160m` (stock) | — | — | baseline |
| `llama3_160m_bitnet_structure_only` | ✗ | ✗ | cost/benefit of the extra pre-norms alone |
| `llama3_160m_bitnet_fp16_weights` | ✓ | ✗ | everything except weight ternarization |
| `llama3_160m_bitnet_no_actquant` | ✗ | ✓ | everything except activation quant |
| `llama3_160m_bitnet` (full) | ✓ | ✓ | the measured +0.13 gap |

Decomposition logic (loss deltas vs stock, additivity not guaranteed but
informative):

- `structure_only − stock` ≈ structural effect of SubLN norms (expected ≈ 0, possibly slightly negative/helpful);
- `fp16_weights − structure_only` ≈ activation-quant cost;
- `no_actquant − structure_only` ≈ weight-ternarization cost;
- `full − fp16_weights` ≈ weight-ternarization cost measured the other way (cross-check);
- `full − no_actquant` ≈ activation-quant cost measured the other way (cross-check).

## Prediction (stated before running)

From the day-1 small-scale results (real ternary gap at tiny scale) and the
BitNet paper's claim that int8 per-token activations are nearly free:

- `structure_only` ≈ stock (within noise, ±0.01);
- `fp16_weights` close to stock — **most of the +0.13 gap disappears** when
  weights stay FP16 (predict ≤ +0.03 of the gap remains);
- `no_actquant` keeps **most of the gap** (predict ≥ +0.09 remains);
- i.e. attribution: weight ternarization is the dominant term.

If instead `fp16_weights` still trails stock by ≥ half the gap, the story is
training dynamics around the act-quant STE / pre-norm interaction, not ternary
weights — that would be the surprising (and more interesting) outcome.

## How to run

Same launch recipe as the MVP comparison (see
[`../../results/README.md`](../../results/README.md)), changing only `--config`
and the dump folder, e.g.:

```bash
CONFIG=llama3_160m_bitnet_fp16_weights   # or _no_actquant / _structure_only
torchrun --nproc_per_node=1 -m torchtitan.train \
  --module torchtitan.models.llama3 --config $CONFIG \
  --training.steps 1700 --training.local-batch-size 16 --training.seq-len 2048 \
  --activation-checkpoint.mode selective --parallelism.data-parallel-shard-degree 1 \
  --metrics.log-freq 10 --validator.enable --validator.freq 170 --validator.steps 20 \
  --checkpoint.enable --checkpoint.interval 850 --checkpoint.keep-latest-k 2 \
  --dataloader.dataset c4 --job.dump-folder ./outputs/$CONFIG
```

Seed-lock note: `debug.seed=42` is committed in the inherited `llama3_160m`
config and the dataloader shuffle seed is hard-coded (42), exactly as in the MVP
runs. All BitNet variants have identical parameter sets (the pre-norms exist in
all of them), so the three ablations + full BitNet are init-identical to each
other; vs stock the param set differs by the pre-norms, as it already did in the
MVP.

## Verification

- Unit tests: `tests/unit_tests/test_bitnet_quantization.py` covers (a) flag-off
  forward paths are exactly `F.linear` / ternary-only, (b) the three registry
  configs convert the same layer set as `llama3_160m_bitnet` with only the quant
  flags differing, and seed/steps/LR matching the MVP config.
- The fingerprint probe
  ([`../2026-06-11-fingerprint-smoke/fingerprint.py`](../2026-06-11-fingerprint-smoke/fingerprint.py))
  can be pointed at any variant to confirm what's actually in the forward path.

## Status

**VARIANTS READY (2026-06-12).** All three ablation configs are implemented,
unit-tested (`tests/unit_tests/test_bitnet_quantization.py`, 6 passed + 1
env-skip on CPU torch 2.11; the registry test runs on the GPU box's nightly),
and CPU-smoke-checked (10 AdamW steps per flag combo, all finite and
decreasing).

## Results

_Pending — GPU worker fills this in. Expected artifacts: per-variant
`*_curves.jsonl` (train.loss / eval.loss / eval.perplexity), final-number table,
and an attribution note comparing against the prediction above._
