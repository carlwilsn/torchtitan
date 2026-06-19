# RESULTS — BitNet 160M 3-seed reproduction

**Post-run analysis. Numbers below are computed by `analyze.py` from the committed
per-run curve JSONLs only (`results/{stock,bitnet}_s{42,43,44}_curves.jsonl`).**
Compare against `PREDICTION.md` (pre-registered, committed before any GPU spend).

Sweep ran on one Lambda A10 (us-east-1), torch `2.12.0.dev20260408+cu128`,
torchtitan commit `0e53726f1`, **2026-06-19 08:51:57Z → 12:58:50Z (4h07m wall clock)**.
All 6 runs exited 0 (193 curve records each — see `results/run_status.txt`,
sentinel `results/SWEEP_DONE.txt`).

## Final metrics at step 1700 (ground truth)

| run | train.loss | eval.loss | eval.ppl |
| --- | ---: | ---: | ---: |
| stock_s42 | 1.4271 | 1.4449 | 4.2414 |
| stock_s43 | 1.4293 | 1.4499 | 4.2625 |
| stock_s44 | 1.3951 | 1.4155 | 4.1184 |
| bitnet_s42 | 1.5398 | 1.5524 | 4.7226 |
| bitnet_s43 | 1.5454 | 1.5584 | 4.7511 |
| bitnet_s44 | 1.5479 | 1.5621 | 4.7689 |

### Per-config mean ± sample-std across 3 seeds

| config | train | val | val ppl |
| --- | ---: | ---: | ---: |
| stock | 1.4172 ± 0.0191 | 1.4367 ± 0.0186 | 4.208 ± 0.078 |
| BitNet | 1.5444 ± 0.0042 | 1.5576 ± 0.0049 | 4.748 ± 0.023 |

### Gap (BitNet − stock), same-seed paired

| metric | s42 | s43 | s44 | **mean** | sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | +0.1127 | +0.1162 | +0.1528 | **+0.1272** | 0.0222 |
| val | +0.1075 | +0.1085 | +0.1467 | **+0.1209** | 0.0223 |
| val ppl | +0.481 | +0.489 | +0.651 | **+0.540** | 0.096 |

## Prediction scorecard — all 5 hold

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| P1 | BitNet val > stock for all 3 seeds (sign never flips) | gaps +0.108 / +0.109 / +0.147, all > 0 | **PASS** |
| P2 | 3-seed mean val gap +0.12, band [0.08, 0.16]; train near +0.13 [0.09, 0.17] | val **+0.1209**, train **+0.1272** | **PASS** |
| P3 | per-config val σ ≈ 0.010 (≤0.025); val-gap σ ≈ 0.015 (≤0.03) | val σ stock 0.0186 / BitNet 0.0049; gap σ 0.0223 | **PASS** |
| P4 | stock ≈ 1.43±0.02, BitNet ≈ 1.56±0.02; no seed diverges/stalls/NaNs | stock 1.437±0.019, BitNet 1.558±0.005, all exit 0 | **PASS** |
| P5 | BitNet ~1.7–1.8× slower, +~1.5 GiB, stable — every seed | stock ~29 min/run, BitNet ~53 min/run = **1.83×**; no spikes | **PASS** |

The pre-registered headline was **+0.12 nat mean val gap**; the measured value is
**+0.1209**. The single-seed anchor (seed 42: train +0.1314 / val +0.1233, commit
`1559a53ce`) sits well within 1σ of the 3-seed mean — it was a representative draw,
not luck. **The ternary-vs-FP16 quality gap at 160M / 55.7M tokens is real,
reproducible, and tight: ≈ +0.12 nat val loss (≈ +0.54 perplexity).**

## Where the seed variance comes from (honest read)

Variance is driven almost entirely by the **stock** side (val σ 0.0186), not BitNet
(val σ 0.0049). Seed 44's larger gap (+0.147) is because stock_s44 trained to the
*lowest* stock loss (1.4155) while BitNet stayed flat — BitNet is the more
seed-stable of the two here. Note the seed scope (from `PREDICTION.md`): the
dataloader interleave seed is fixed at 42, so `--debug.seed N` varies model-init +
optimization RNG with **data order held constant**. This σ is therefore a *lower
bound* on full run-to-run spread (data-resampling variance would be somewhat larger).

## Reproduce

```bash
# torchtitan @ 0e53726f1, torch 2.12.0.dev20260408+cu128, 1x A10
bash run_3seed_sweep.sh          # launches all 6 runs sequentially, writes results/
python analyze.py                # regenerates every number above from the JSONLs
```

## Cost / wall-clock honesty

- **Compute:** 4h07m for all 6 runs (stock ~29 min, BitNet ~53 min each) ≈ **$5.3** @ $1.29/hr.
- **Cost-integrity miss (recorded, not hidden):** the run-watcher stopped polling
  before the `SWEEP_DONE.txt` sentinel was written, so teardown was not triggered at
  completion (12:58Z). The box sat **idle ~6 h** before termination (~18:55Z),
  wasting ≈ **$7.7**. Artifacts were never at risk (committed+pushed to origin/main
  by the run worker at completion); the loss was pure idle spend. Lesson logged to the
  goal file: a WatchRun whose `check_command` can silently stop emitting must be paired
  with a time-boxed backup teardown wake, not trusted as the sole done-trigger.
- **Final state:** box terminated, `lambda list` → `{"instances": []}` (verified).
