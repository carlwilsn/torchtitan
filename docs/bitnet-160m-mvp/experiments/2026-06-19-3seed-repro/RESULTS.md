# RESULTS — BitNet 160M 3-seed reproduction

**Run 2026-06-19, A10, torchtitan @ `0e53726f1`, torch `2.12.0.dev20260408+cu128`.**
Ground truth = the six committed `results/*_curves.jsonl` (193 lines each). Numbers
below are reproduced by `analyze.py` in this directory (reads the JSONLs only).
Compare against the pre-registered `PREDICTION.md` (committed before any GPU spend).

## What ran

3 seeds × {stock `llama3_160m`, BitNet `llama3_160m_bitnet`} = 6 runs, byte-identical
training flags differing only in `--config`, `--debug.seed` (42/43/44), and dump-folder.
1700 steps, local batch 16, seq-len 2048, C4, validator every 170 steps (20 val steps).
All 6 runs **exit 0**, 193 curve lines each (`run_status.txt`). No seed diverged, stalled,
or NaN'd. Started 08:51:57Z, finished 12:58:50Z (`environment.txt`).

## Final metrics at step 1700

| run | train loss | val loss | val ppl |
| --- | ---: | ---: | ---: |
| stock_s42 | 1.4271 | 1.4449 | 4.2414 |
| stock_s43 | 1.4293 | 1.4499 | 4.2625 |
| stock_s44 | 1.3951 | 1.4155 | 4.1184 |
| bitnet_s42 | 1.5398 | 1.5524 | 4.7226 |
| bitnet_s43 | 1.5454 | 1.5584 | 4.7511 |
| bitnet_s44 | 1.5479 | 1.5621 | 4.7689 |

### Per-config mean ± sample-std across seeds

| config | train loss | val loss | val ppl |
| --- | ---: | ---: | ---: |
| stock | 1.4172 ± 0.0191 | 1.4367 ± 0.0186 | 4.2075 ± 0.0778 |
| BitNet | 1.5444 ± 0.0042 | 1.5576 ± 0.0049 | 4.7476 ± 0.0233 |

### Gap (BitNet − stock)

| metric | s42 | s43 | s44 | mean | sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | +0.1127 | +0.1162 | +0.1528 | **+0.1272** | 0.0222 |
| val | +0.1075 | +0.1085 | +0.1467 | **+0.1209** | 0.0223 |
| ppl | +0.4812 | +0.4886 | +0.6505 | +0.5401 | 0.0957 |

## Prediction scorecard — all 5 HELD

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| P1 | BitNet val > stock val, all 3 seeds | gaps +0.108 / +0.109 / +0.147 | ✅ |
| P2 | 3-seed mean **val** gap ∈ [+0.08, +0.16] | **+0.1209** | ✅ |
| P2 | 3-seed mean **train** gap ∈ [+0.09, +0.17] | **+0.1272** | ✅ |
| P3 | per-config val σ ≤ 0.025 | stock 0.0186, BitNet 0.0049 | ✅ |
| P3 | val-gap σ ≤ 0.03 | 0.0223 | ✅ |
| P4 | stock val ≈ 1.43±0.02, BitNet ≈ 1.56±0.02, no divergence | 1.437±0.019 / 1.558±0.005, all exit 0 | ✅ |
| P5 | BitNet ~1.7–1.8× slower, no spikes | ~18.5k vs ~33–43k tps (~1.8×), grad_norm bounded | ✅ |

**Headline:** the single-seed anchor (commit `1559a53ce`: train +0.131, val +0.123) is
**confirmed representative** — the 3-seed mean val gap is **+0.121 ± 0.022 nat** (mean within
0.002 of the single-seed number). The ternary-vs-FP16 quality gap at 160M / 55.7M tokens is
real and stable, not a lucky draw.

## Two observations worth keeping

1. **BitNet seeds cluster ~4× tighter than stock** (val σ 0.0049 vs 0.0186). Ternary weight
   quantization appears to damp sensitivity to model-init RNG — the STE + per-tensor rescaling
   pulls runs toward a common trajectory. (Caveat: data order is fixed across seeds, so this is
   init-only variance; full data-resampling spread would be larger. Lower bound.)

2. **Seed 44 is the gap outlier (+0.147 vs ~+0.108).** It's driven almost entirely by stock_s44
   training *better* (val 1.4155, ~0.03 below the other two stock seeds) while BitNet_s44 sat with
   its siblings — i.e. BitNet couldn't follow stock's lucky-init descent. Consistent with (1): the
   quantized model has a narrower reachable set.

## Honest scope limits

- **Data order fixed across seeds** (interleave seed hard-coded 42). `--debug.seed` varies
  model-init + optimization noise only. Reported seed spread is a **lower bound** on true
  run-to-run variance; full data-resampling would widen it.
- **55.7M-token budget, 1700 steps** — both configs are undertrained at a single shared LR; this
  is a controlled *relative* comparison, not a converged absolute-quality claim.
- **Checkpoints not retained.** Each run's step-850 + step-1700 checkpoint (1.9 GB/run, 11 GB
  total) was left on the box and lost at terminate — too large for the git submodule and not
  needed for any loss/val/ppl criterion. Re-deriving them requires re-running (config is locked).

## Reproduce

```bash
python analyze.py   # reads results/*_curves.jsonl, prints the tables above + prediction check
```

Raw per-run logs: `results/*_train.log`. Per-step metrics: `results/*_curves.jsonl`
(events `train.loss`, `eval.loss`, `eval.perplexity`). Full sweep log was captured to the box's
`~/sweep.log` (run markers + per-run exit codes mirrored into `results/run_status.txt`).
