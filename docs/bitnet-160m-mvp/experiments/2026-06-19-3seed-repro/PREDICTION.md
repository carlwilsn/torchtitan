# PRE-REGISTERED PREDICTION — BitNet 160M 3-seed reproduction

**Written 2026-06-19, BEFORE any GPU spend for this sweep.** This file is committed
to git before the runs launch so the prediction cannot be edited post-hoc. The
post-run analysis (`RESULTS.md`) compares against the numbers stated here verbatim.

## What we are running

3 seeds × {stock `llama3_160m`, BitNet `llama3_160m_bitnet`} = **6 runs**, byte-identical
training flags (see `run_3seed_sweep.sh`), differing only in `--config` and `--debug.seed`
(42, 43, 44) and the dump-folder. This is a direct multi-seed extension of the single-seed
1700-step run committed at `1559a53ce` (analysis: `../../learning-docs/12-1700-step-seedlocked-comparison.md`).

**Seed scope (honest):** the dataloader interleave/shuffle seed is hard-coded to 42 in the
config, so `--debug.seed N` varies the **model-init RNG** (and downstream optimization noise),
while the **data order is held fixed across all seeds**. Same-seed stock/BitNet therefore see
identical data — the fairness property. The seed-to-seed variance we measure is therefore
*init + optimization* variance under a shared data order, **not** full data-resampling variance.
We expect data-resampling variance would be somewhat larger; this is a lower bound on total
run-to-run spread.

## The single-seed anchor (seed 42, commit 1559a53ce)

| Quantity | Stock | BitNet | Gap (BitNet − stock) |
| --- | ---: | ---: | ---: |
| Final train loss (step 1700) | 1.4123 | 1.5438 | **+0.1314 nat** |
| Final val loss (step 1700) | 1.4342 | 1.5576 | **+0.1233 nat** |
| Final val perplexity | 4.1965 | 4.7473 | +0.55 |

## Predictions (the falsifiable claims)

**P1 — Direction is robust.** BitNet final val loss > stock final val loss for **all 3 seeds**.
The gap is positive every seed (no seed flips the sign).

**P2 — Magnitude.** The **3-seed mean val gap** lands at **+0.12 nat**, in the band
**[+0.08, +0.16]**. (Centered on the single-seed +0.123; band ±0.04 absorbs seed spread.)
The mean **train gap** lands near **+0.13 nat**, band [+0.09, +0.17].

**P3 — Seed-to-seed variance is small (fixed data order).**
- Per-config final val-loss standard deviation across the 3 seeds: **σ ≈ 0.010 nat**, expected
  range [0.003, 0.025]. (Init-only variance at 160M / 55.7M tokens is small.)
- Standard deviation of the **val gap** across seeds: **σ_gap ≈ 0.015 nat**, expected ≤ 0.03.
  → the single-seed +0.123 is **representative**, not a lucky/unlucky draw: it should sit within
  ~1σ of the 3-seed mean.

**P4 — Absolute losses cluster tightly.** Stock final val loss ≈ 1.43 ± 0.02 across seeds;
BitNet ≈ 1.56 ± 0.02. Both clearly below the shakedown noise floor; no seed diverges or stalls.

**P5 — Cost/behavior unchanged by seed.** BitNet ~1.7–1.8× slower than stock, +~1.5 GiB peak
memory, grad_norm bounded (stock tail ~0.5–0.7, BitNet ~1.4–1.6), no unmirrored loss spikes —
same per every seed. The U-shaped gap curve (shrinks ~step 850, reopens, plateaus) reappears.

## What would FALSIFY this

- A seed where BitNet val loss ≤ stock (falsifies P1).
- 3-seed mean val gap outside [+0.08, +0.16] (falsifies P2).
- Gap σ across seeds > 0.03 nat, i.e. the gap swings wildly seed-to-seed (falsifies P3 — would
  mean the single-seed number was not trustworthy and the "+0.13" headline needs widening).
- Any seed that diverges, stalls, or NaNs (falsifies P4 — would be a stability finding).

## Rationale (one paragraph)

At 160M params / 55.7M tokens on streaming c4 with a fixed data order, the dominant remaining
randomness is weight-init RNG. Large-LM training is famously low-variance across init seeds at
fixed data once past the early transient, so we expect per-config σ on the order of 0.01 nat.
The stock-vs-BitNet gap is a *structural* (representational-capacity) effect — ternary weights
carry 3 levels × one per-tensor scale — so it should be far more stable across seeds than the
absolute loss, reproducing the single-seed +0.12 nat closely. The honest risk is that 0.01 nat
is optimistic and we see ~0.02–0.025; that still leaves the gap (~0.12) an order of magnitude
above the spread, which is the whole point of running 3 seeds: to show the gap is real, not noise.
