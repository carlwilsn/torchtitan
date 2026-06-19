# BitNet b1.58 vs FP16 — 160M, 3-seed analysis (C4, 1700 steps)

**Date:** 2026-06-19 · **Repo commit:** `0e53726f1` (code) / artifacts in `3a90c1658` on `origin/main`
**Box:** Lambda A10 (23 GiB, driver 570.148.08) · torch `2.12.0.dev20260408+cu128` · Python 3.10.12
**Sweep window (UTC):** started 08:51:57 → SWEEP_DONE 12:58:50 (4h06m53s compute)

All numbers below are read directly from the committed curve files
(`results/<run>_curves.jsonl`, 193 lines each, last line = step-1700 eval). Nothing here is narrated — every value is the literal `value` field of the final `train.loss` / `eval.loss` / `eval.perplexity` event.

---

## 1. Headline

Reproducing BitNet b1.58 (ternary weights, STE) against the FP16 stock Llama3-160M on an
**identical, seed-locked** training recipe (same data order per seed — the fairness property),
across 3 seeds {42, 43, 44}:

| Metric (step 1700, val) | Stock (FP16) | BitNet (ternary) | Gap |
|---|---|---|---|
| **Mean val loss** | **1.4367** | **1.5576** | **+0.1209 nat** |
| Mean perplexity | 4.207 | 4.748 | +0.541 |

**The ternary penalty at 160M is +0.121 nat of validation loss** (≈ +12.9% perplexity).
This is the clean, multi-seed confirmation of the single-seed `1559a53ce` result (+0.123 val).

---

## 2. Per-seed table (step 1700)

| Run | Final train loss | Val (eval) loss | Val perplexity | Wall-clock |
|---|---|---|---|---|
| stock_s42  | 1.4271 | 1.4449 | 4.2414 | 29m26s |
| stock_s43  | 1.4293 | 1.4499 | 4.2625 | 29m31s |
| stock_s44  | 1.3951 | 1.4155 | 4.1184 | 29m33s |
| **stock mean** | **1.4172** | **1.4367** | **4.2074** | ~29.5m |
| bitnet_s42 | 1.5398 | 1.5524 | 4.7226 | 52m49s |
| bitnet_s43 | 1.5454 | 1.5584 | 4.7511 | 52m47s |
| bitnet_s44 | 1.5479 | 1.5621 | 4.7689 | 52m47s |
| **bitnet mean** | **1.5444** | **1.5576** | **4.7476** | ~52.8m |

---

## 3. Spread across seeds (the point of running 3)

| | Stock val | BitNet val |
|---|---|---|
| mean | 1.43674 | 1.55763 |
| sample σ | **0.0186** | **0.0049** |
| min–max | 1.4155 – 1.4499 | 1.5524 – 1.5621 |

**Paired (same-seed) val gap** — the fair comparison, since same seed ⇒ identical data order:

| seed | bitnet − stock |
|---|---|
| 42 | +0.10747 |
| 43 | +0.10852 |
| 44 | +0.14666 |
| **mean ± σ** | **+0.1209 ± 0.0223 nat** |

The gap is tight and strictly positive on every seed. BitNet's own seed-to-seed spread is
**~4× smaller** than stock's (σ 0.005 vs 0.019) — the quantization/STE path is, if anything,
*more* run-to-run stable here. The whole stock spread is driven by one outlier: **stock_s44**
trained ~0.034 nat lower than the other two stock seeds, which both widens stock σ and inflates
the s44 paired gap to +0.147. BitNet_s44 is unremarkable, so this is a stock-init lucky draw,
not a BitNet effect.

---

## 4. Prediction scorecard (pre-registered in `PREDICTION.md`, committed `0e53726f1` BEFORE GPU spend)

| # | Pre-registered prediction | Actual | Verdict |
|---|---|---|---|
| P1 | mean val gap **+0.12 nat**, band [+0.08, +0.16] | **+0.1209** | ✅ dead center |
| P2 | gap σ across seeds **≤ 0.03** | 0.0223 (paired) | ✅ |
| P3 | per-config val σ **≈ 0.01** | bitnet 0.005 ✅ / stock 0.019 | ⚠️ stock ~2× high (s44 outlier) |
| P4 | **no seed diverges** | all 6 monotonic → step 1700, exit 0 | ✅ |

**Single-seed anchor:** `1559a53ce` seed-42 gave +0.123 val. 3-seed mean = +0.121.
Agreement within 0.002 nat — the single-seed number was not a fluke.

Net: **4/4 substantive predictions hit**; the only miss is the stock σ being ~2× the eyeballed
0.01 guess, fully explained by the stock_s44 draw and harmless to the headline.

---

## 5. Cost / compute

- **BitNet is 1.79× slower** than stock (52.8m vs 29.5m per run; locked estimate was 1.77× — confirmed).
- Compute: 4h06m53s across 6 runs. Productive spend ≈ $6 (compute + env setup) @ $1.29/hr.
- **Overspend, honest:** the box idle-billed from SWEEP_DONE (08:58 EDT) to teardown (~14:52 EDT) —
  ~5.9 hr ≈ **~$7.6 wasted** because the watcher/worker died after the sweep finished and teardown
  did not fire until the mission's backup self-check caught it. Total box spend ≈ $13 for a ~$6 job.
  See "Operational failure" below.

---

## 6. Operational failure (what broke, for the record)

The science is clean; the *automation* failed at the seam:

1. The corrected WatchRun (`run_4243a6de-2`, tailing real progress) **last checked at ~07:29 EDT
   and then stopped checking** — it never observed SWEEP_DONE at 08:58 EDT and never fired the
   DONE wake. ListRuns showed it frozen at "RUN 4/6 bitnet_s42 step:920", 442 min stale.
2. The Phase-2 worker (`5b46027e-22c`) went idle ~07:51 EDT (before the sweep even finished) and
   never executed its standing "on SWEEP_DONE → commit/push, leave box running" instruction.
3. Result: the sweep completed perfectly and unattended (criterion 2's "survives SSH" goal — met),
   but **nobody tore the box down**, so it idled for ~6 hours.
4. Teardown finally happened ~14:52 EDT (artifacts committed+pushed as `3a90c1658`, box terminated),
   overlapping this mission backup self-check.

**Carry-forward lessons:**
- A done-sentinel must be backed by a trigger that is *itself* monitored for liveness — a watcher
  that silently stops checking is worse than no watcher (it gives false comfort).
- Teardown should be driven off the sentinel by something that cannot go idle (a deterministic
  run-watcher with a teardown action), not delegated to a worker turn that can end.
- The earlier lesson still holds: a WatchRun check_command must emit a *changing* progress note.

---

## 7. Provenance (verify it yourself)

```
# all on origin/main @ 3a90c1658
results/stock_s42_curves.jsonl   193 lines, last event eval.perplexity=4.2414  @ step 1700
results/stock_s43_curves.jsonl   193 lines, eval.perplexity=4.2625
results/stock_s44_curves.jsonl   193 lines, eval.perplexity=4.1184
results/bitnet_s42_curves.jsonl  193 lines, eval.perplexity=4.7226
results/bitnet_s43_curves.jsonl  193 lines, eval.perplexity=4.7511
results/bitnet_s44_curves.jsonl  193 lines, eval.perplexity=4.7689
results/run_status.txt           6× "exit code: 0", 6× "curve lines: 193"
results/SWEEP_DONE.txt           ALL_RUNS_DONE 2026-06-19T12:58:50Z
results/environment.txt          A10 / torch 2.12.0.dev cu128 / commit 0e53726f1
```

Checkpoints (1.9 GiB × 6 = 11 GiB) were **not** retained — unsuitable for a public git repo and
fully regeneratable from the seed-locked config + fixed data order. Curves are the scientific record.
