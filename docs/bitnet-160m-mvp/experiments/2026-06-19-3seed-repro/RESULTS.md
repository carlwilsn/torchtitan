# RESULTS — BitNet 160M 3-seed reproduction

**Run 2026-06-19, A10 (us-east-1), torchtitan@`0e53726f1`, torch `2.12.0.dev20260408+cu128`.**
Compared verbatim against the pre-registered `PREDICTION.md` (committed before any GPU spend).
All numbers below are read directly from the committed `results/*_curves.jsonl` (the only trusted evidence).

## What ran

3 seeds (42, 43, 44) × {stock `llama3_160m`, BitNet `llama3_160m_bitnet`} = **6 runs**, byte-identical
flags differing only in `--config`, `--debug.seed`, and dump-folder. 1700 steps, local-batch 16,
seq-len 2048, c4 dataset, validator every 170 steps × 20 steps. Data order held fixed across seeds
(dataloader interleave seed hard-coded 42) → measured spread is **init + optimization** variance under
a shared data order, a lower bound on full run-to-run variance.

**All 6 runs exited 0, 193 curve lines each** (`run_status.txt`). No divergence, stall, or NaN.
Wall clock: 08:51:57Z → 12:58:50Z = **4h07m compute**. Stock ~29 min/run, BitNet ~48–52 min/run (~1.66×).

## Final metrics (step 1700)

| run | train loss | val loss | val ppl |
| --- | ---: | ---: | ---: |
| stock_s42 | 1.4271 | 1.4449 | 4.2414 |
| stock_s43 | 1.4293 | 1.4499 | 4.2625 |
| stock_s44 | 1.3951 | 1.4155 | 4.1184 |
| bitnet_s42 | 1.5398 | 1.5524 | 4.7226 |
| bitnet_s43 | 1.5454 | 1.5584 | 4.7511 |
| bitnet_s44 | 1.5479 | 1.5621 | 4.7689 |

### Per-config aggregate (mean ± population σ across 3 seeds)

| config | train loss | val loss | val ppl |
| --- | ---: | ---: | ---: |
| stock | 1.4172 ± 0.0156 | 1.4367 ± 0.0152 | 4.207 |
| BitNet | 1.5444 ± 0.0034 | 1.5576 ± 0.0040 | 4.748 |

### The gap (BitNet − stock)

| seed | val gap | train gap | ppl gap |
| --- | ---: | ---: | ---: |
| 42 | +0.1075 | +0.1127 | +0.481 |
| 43 | +0.1085 | +0.1162 | +0.489 |
| 44 | +0.1467 | +0.1528 | +0.651 |
| **mean** | **+0.1209** | **+0.1272** | **+0.540** |
| σ (pop / sample) | 0.0182 / 0.0223 | 0.0181 / 0.0222 | — |

## Prediction scorecard

| # | Claim | Predicted | Observed | Verdict |
| --- | --- | --- | --- | --- |
| P1 | Direction robust (all seeds gap>0) | all positive | +0.108, +0.109, +0.147 | ✅ PASS |
| P2 | Mean **val** gap | +0.12, band [0.08, 0.16] | **+0.1209** | ✅ PASS (near-exact) |
| P2 | Mean **train** gap | +0.13, band [0.09, 0.17] | +0.1272 | ✅ PASS |
| P3 | Gap σ across seeds | ≈0.015, ≤0.03 | 0.018 pop / 0.022 sample | ✅ PASS |
| P3 | Per-config val σ | ≈0.010, [0.003, 0.025] | stock 0.015, BitNet 0.004 | ✅ in band |
| P4 | Absolute losses cluster, no divergence | stock 1.43±.02, bitnet 1.56±.02 | 1.437 / 1.558, all exit 0 | ✅ PASS |
| P5 | BitNet ~1.7–1.8× slower | 1.7–1.8× | ~1.66× | ✅ ~PASS (slightly faster) |

**No prediction was falsified.** The single-seed anchor (+0.123 val, commit `1559a53ce`) sits
within ~1σ of the 3-seed mean — it was representative, not a lucky draw. This is the point of
running 3 seeds: the gap (~0.12 nat) is **an order of magnitude larger than its seed spread (~0.018)**,
so it is a real structural effect, not noise.

## The one genuine finding beyond the prediction

**BitNet's across-seed variance is ~4× tighter than stock's** (val σ 0.004 vs 0.015). Seed 44
drove the only notable gap-widening (+0.147 vs ~0.108): *stock_s44* got a lucky-good init
(val 1.4155, the best of any stock run) while *bitnet_s44* stayed mid-pack with the other BitNet
seeds. So the wider gap at s44 is a **stock** outlier, not a BitNet one. Interpretation: the ternary
weight constraint (3 levels × one per-tensor scale) acts as a regularizer/variance-reducer on the
final-loss init sensitivity. Worth a follow-up — it predicts BitNet should be *more* reproducible
seed-to-seed than full-precision at this scale.

## Caveats / scope (honest)

- Fixed data order → this is **init+optim** variance, a lower bound on total run-to-run spread.
  Full data-resampling variance would likely be larger; don't over-read the tight BitNet σ.
- 160M params / 55.7M tokens is **severely undertrained** (single shared LR, short budget). The gap
  here is the undertrained-regime gap, consistent with the earlier width sweep's peak-at-160M shape.
  It is **not** a converged-model claim.
- Perplexity here is torchtitan's c4-val perplexity at this budget, not a standard benchmark ppl.

## Artifacts (all committed, commit `a08d6963c`, pushed to origin/main)

- `results/{stock,bitnet}_s{42,43,44}_curves.jsonl` — full per-step metrics (193 lines each)
- `results/*_train.log` — human-readable console logs (force-added past `*.log` ignore)
- `results/run_status.txt` — exit codes + line counts; `results/environment.txt` — GPU/torch/commit/timing
- `results/SWEEP_DONE.txt` — `ALL_RUNS_DONE 2026-06-19T12:58:50Z` sentinel
- `results/val_curves_and_gap.png` — val curves (solid stock / dashed BitNet) + gap-vs-step per seed
- **Checkpoints (1.9 GB × 6 = 11.4 GB) intentionally NOT committed** — disposable 1700-step scratch,
  too large for GitHub, and the gap verdict lives entirely in the curves. Box wiped on terminate.
