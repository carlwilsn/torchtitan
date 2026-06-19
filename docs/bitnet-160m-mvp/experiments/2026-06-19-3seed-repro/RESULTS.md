# RESULTS — BitNet 160M 3-seed reproduction

**Run:** 2026-06-19, Lambda A10 (`bitnet-3seed`, us-east-1), torchtitan@`0e53726f1`.
**Compares against:** the pre-registered, pre-spend predictions in [`PREDICTION.md`](./PREDICTION.md) (committed before any GPU spend) and the single-seed anchor at commit `1559a53ce`.
**Raw artifacts:** [`results/`](./results/) — 6 × `*_curves.jsonl` (193 metric lines each), `run_status.txt`, `environment.txt`, `SWEEP_DONE.txt`. All verified against the raw files, not narration.

## What ran

3 seeds (42, 43, 44) × {stock `llama3_160m`, BitNet `llama3_160m_bitnet`} = **6 runs**, byte-identical training flags (see [`run_3seed_sweep.sh`](./run_3seed_sweep.sh)), differing only in `--config`, `--debug.seed`, and dump-folder. 1700 steps, batch 16, seq-len 2048, c4, selective activation checkpointing, degree-1 FSDP. All 6 completed **exit code 0**, 193 curve lines each — no seed diverged, stalled, or NaN'd.

> **Seed-scope caveat (carried from PREDICTION):** the dataloader interleave seed is hard-coded to 42, so `--debug.seed N` varies **model-init RNG + optimization noise only**; data order is held fixed across all seeds. Same-seed stock/BitNet see identical data (the fairness property). The measured seed spread is a **lower bound** on full run-to-run variance (no data resampling).

## Final metrics (step 1700, from raw curves)

| run | train loss | val loss | val ppl |
| --- | ---: | ---: | ---: |
| stock_s42 | 1.4271 | 1.4449 | 4.241 |
| stock_s43 | 1.4293 | 1.4499 | 4.263 |
| stock_s44 | 1.3951 | 1.4155 | 4.118 |
| bitnet_s42 | 1.5398 | 1.5524 | 4.723 |
| bitnet_s43 | 1.5454 | 1.5584 | 4.751 |
| bitnet_s44 | 1.5479 | 1.5621 | 4.769 |

**Aggregates (mean ± population σ across 3 seeds):**

| config | train loss | val loss | val ppl |
| --- | ---: | ---: | ---: |
| stock | 1.4172 ± 0.0156 | **1.4367 ± 0.0152** | 4.208 ± 0.064 |
| BitNet | 1.5444 ± 0.0034 | **1.5576 ± 0.0040** | 4.748 ± 0.019 |

**The headline — ternary-vs-FP16 gap:**

| gap (BitNet − stock) | per-seed (42, 43, 44) | mean ± σ |
| --- | --- | ---: |
| val loss | +0.1075, +0.1085, +0.1467 | **+0.1209 ± 0.0182 nat** |
| train loss | +0.1127, +0.1162, +0.1528 | +0.1272 ± 0.0181 nat |

The gap is **positive on every seed** and its mean (+0.121 nat val) sits an order of magnitude above its own seed spread (σ 0.018) — the gap is a real structural effect, not seed noise. In perplexity terms: BitNet ppl 4.75 vs stock 4.21, a **+0.54 ppl** penalty for ternary weights.

## Prediction scorecard — 5 / 5 confirmed

| # | Pre-registered claim | Result | Verdict |
| --- | --- | --- | --- |
| **P1** | BitNet val > stock for all 3 seeds (sign never flips) | all 3 gaps positive (+0.108, +0.109, +0.147) | ✅ |
| **P2** | mean val gap +0.12 nat, band [+0.08, +0.16]; train gap ~+0.13, [+0.09,+0.17] | val **+0.121**, train **+0.127** | ✅ dead-center |
| **P3** | per-config val σ ≈ 0.010 (range [0.003,0.025]); gap σ ≈ 0.015 (≤0.03) | val σ: stock 0.015, BitNet 0.004; **gap σ 0.018** | ✅ within bands |
| **P4** | stock val ≈ 1.43±0.02, BitNet ≈ 1.56±0.02; no divergence | stock 1.437±0.015, BitNet 1.558±0.004; 0 diverged | ✅ |
| **P5** | BitNet ~1.7–1.8× slower, +~1.5 GiB, grad_norm bounded, no spikes | **1.80× slower**, **+1.47 GiB**, bounded grads, no spikes | ✅ (see note) |

**P5 detail (from train logs):** stock ~32,950 tps / grad_norm ~0.35–0.38 / 9.46 GiB; BitNet ~18,500 tps / grad_norm ~0.66–0.75 / 10.93 GiB. Throughput ratio 1.80× and memory delta +1.47 GiB match the single-seed anchor exactly. **One honest sub-miss:** I predicted BitNet's grad_norm tail at 1.4–1.6, but it ran *calmer* (~0.66) — gradients were better-behaved than expected. The qualitative claim (bounded, no unmirrored spikes) holds; the magnitude guess was off.

## Reproduction verdict

The single-seed +0.123 nat val gap (commit `1559a53ce`) **reproduces** as a 3-seed mean of **+0.121 ± 0.018 nat**. The single-seed number was representative, not a lucky draw: its seed (s42) gave +0.108, within 1σ of the 3-seed mean. The "+0.12 nat ternary penalty at 160M / 55.7M tokens" headline is now backed by 3 seeds with the gap >> spread.

One texture worth noting: **stock's seed spread (σ 0.015) is ~4× BitNet's (σ 0.004).** BitNet's per-tensor ternary quantization appears to *regularize* run-to-run init variance — the ternary models land in a tighter band. This wasn't predicted and is a candidate Open Question (is the tighter BitNet spread a real quantization-as-regularizer effect, or an artifact of fixed data order at this scale?).

## Cost & wall-clock (honest accounting)

| | |
| --- | --- |
| Productive compute (6 runs) | **4.11 hr** (08:51:57Z → 12:58:50Z) |
| Per-run | stock ~29.4 min, BitNet ~52.8 min (1.80×) |
| Box uptime (launch → terminate) | ~10.2 hr |
| **Idle-after-completion** | **~5.95 hr** ($1.29/hr ≈ **$7.7 leaked**) |
| Total box cost | ~$13.2 |

**Cost-integrity miss (reported, not hidden):** the sweep finished 12:58 UTC (08:58 EDT) but the box was not terminated until ~14:55 EDT — it idled ~6 hr. Root cause: the `WatchRun` watcher (`run_4243a6de-2`) went **stale** — it stopped checking and reported frozen progress instead of firing a DONE wake to the mission thread. Teardown was ultimately triggered by the scheduled backup self-check, not the watcher. **Lesson:** do not rely on `WatchRun` alone to wake on completion; pair every unattended run with a time-boxed backup self-check, and have the run script's own completion path (or a cron) terminate the box. The science is unaffected (all artifacts complete + committed before terminate); the loss is ~$7.7.

## Artifacts & provenance

- Curves + status committed to `origin/main` at `3a90c1658` (torchtitan, github.com/carlwilsn/torchtitan): `docs/bitnet-160m-mvp/experiments/2026-06-19-3seed-repro/results/`.
- `environment.txt`: torch `2.12.0.dev20260408+cu128`, A10 23 GB, commit `0e53726f1`.
- Raw `*_train.log` (gitignored, too large) and 11 GB of checkpoints stayed on the box — reproducible from seed + locked config; not worth storing.
- Box `bitnet-3seed` terminated 2026-06-19 (lifecycle tool: `status: terminating`, filesystem wiped after push). `lambda list` → no active instances.
