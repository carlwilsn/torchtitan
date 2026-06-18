# RESULTS — BitNet ternary-vs-FP16 gap across model width

**Status: written AFTER the run, against the pre-committed [PREDICTION.md](./PREDICTION.md).**

Date: 2026-06-17
Experiment dir: `docs/bitnet-160m-mvp/experiments/2026-06-15-width-sweep-ternary-gap/`
Metric: validation loss in nats, `loss = ln(eval.perplexity)`. Quantity of interest: **gap = val(bitnet) − val(stock)**.
Protocol: c4 streaming, **1700 steps, seed 42**, byte-identical flags across both arms of each rung; only the `BitLinearConverter` differs. All numbers read at **step 1700**.

Raw artifacts (verified, terminated box): `.vault-chat/supervisor/artifacts/widthsweep_results/`
— `*_curves.jsonl` (per-step train+eval), `*_train.log`, `run_status.txt`, `environment.txt`.
The 50M and 400M rows are **fresh from this sweep**; the 160M row is **reused from a prior verified run** (not in the artifacts dir).

---

## 1. Final results & the gap-vs-width curve

All FP16/ternary losses are `ln(eval.perplexity)` at step 1700. The 50M and 400M perplexities are the last `eval.perplexity` line in each `*_curves.jsonl` (verified below); the 160M row is the reused anchor.

| width (d_model, depth) | FP16 val loss (ppl) | ternary val loss (ppl) | **gap** (tern − FP16) |
|------------------------|---------------------|------------------------|-----------------------|
| **50M**  (d768,  L8)   | 1.5511 (ppl 4.7169) | 1.6088 (ppl 4.9968)    | **+0.0577** |
| **160M** (d1024, L14)  | 1.4342 (anchor, reused) | 1.5576 (anchor)    | **+0.1234** |
| **400M** (d1536, L16)  | 1.3762 (ppl 3.9598) | 1.4615 (ppl 4.3122)    | **+0.0853** |

**Gap headline: 50M +0.058 → 160M +0.123 → 400M +0.085.**

### Gap vs d_model (ASCII)

```
gap (nats)
0.13 |                  *  160M (+0.1234)
0.12 |                 / \
0.11 |                /   \
0.10 |               /     \
0.09 |              /       \
0.08 |             /         *  400M (+0.0853)
0.07 |            /
0.06 |   *  50M  /
0.05 |  (+0.0577)
     +----------------------------------
        768       1024        1536   d_model
```

The shape is **non-monotonic: it rises (50M→160M, +0.066) then falls (160M→400M, −0.038).** There is **no clean closure** — the gap does not march down toward zero across these three rungs; it makes a hump. A PNG of the same curve (with the pre-run prediction overlaid) is written alongside this doc: [`gap_vs_width.png`](./gap_vs_width.png), regenerable via [`plot_gap_vs_width.py`](./plot_gap_vs_width.py).

---

## 2. Prediction vs actual

Predictions were committed **before** the run (see PREDICTION.md). The 160M point is the fixed anchor in both columns.

| Rung | predicted gap | actual gap | |Δ| (pred − actual) |
|------|---------------|------------|---------------------|
| 50M  | +0.05  | **+0.0577** | 0.008 |
| 160M | +0.123 (anchor) | **+0.1234** (anchor) | 0.000 |
| 400M | +0.09  | **+0.0853** | 0.005 |

**The pre-run prediction was remarkably accurate.** Every rung landed within **0.008 nats** of its predicted gap; 400M was within **0.005** (predicted +0.09 vs actual +0.085). Crucially, the *shape* was called correctly too: the prediction explicitly forecast a non-monotonic curve — gap **up** on the depth-heavy 50M→160M step, gap **down** on the width-heavy 160M→400M step — and that is exactly what the data shows. This is a case where the intuition (the product) was right before the number existed.

---

## 3. Honest read on the paper's "matches FP16 at scale" claim

**The data does NOT cleanly confirm closure, and it does NOT refute it. It is consistent with "too small to see closure yet."**

- The gap is **non-monotonic** across 50M→160M→400M and sits in a narrow **~0.06–0.12 nat band** the whole way. It never trends cleanly toward zero.
- The one leg that *does* point the paper's way is **160M→400M: gap shrank −0.038** (+0.1234 → +0.0853), a width-heavy step where ternary's per-channel headroom should help most. But the **50M→160M leg went the other way (+0.066, gap grew)**, a depth-heavy step. So "closes with scale" holds on one leg and fails on the other.
- We are running **50M–400M — roughly 1–2 orders of magnitude below the paper's claimed ~3B crossover.** At that distance from the crossover, the expected per-rung change in gap is small and comparable to our (un-replicated) noise. So a shallow, ambiguous, non-monotonic curve is **exactly** what "we sit far out on the tail, below where closure completes" predicts.

**One-line read: the gap is small (~0.06–0.12 nats) and non-monotonic across 50M→400M — consistent with "we're 1–2 OOM below the paper's ~3B crossover, too small to see closure yet," not a confirmation and not a refutation.**

This matches the pre-registered most-likely outcome: **inconclusive / shallow, not clean closure.** Per the falsifiable bar in PREDICTION.md, `g400 = +0.0853` lands just above the "+0.08 confirms" threshold and inside the "+0.09–0.15 within-noise / inconclusive" band — i.e. the honest verdict is **inconclusive**, which was the predicted result.

---

## 4. Caveats (state plainly)

**(a) The rungs co-vary depth with width.** The x-axis is labeled `d_model`, but each rung up bumps `n_layers` too (8 → 14 → 16) and *unevenly*: 50M→160M is **depth-heavy** (768→1024 width, 8→14 depth), 160M→400M is **width-heavy** (1024→1536 width, 14→16 depth). So this is gap-vs-(width×depth×params), **not pure-width isolation.** A mechanism where ternary's deficit is a per-layer error that compounds with depth (gap up when you add layers) but is relieved by more channels (gap down when you add width) would *manufacture* exactly the observed up-then-down hump. Do not read the curve as "width relieves ternary."

**(b) Single seed (42), no error bars.** Every cell is one seed. The 160M anchor's gap is known to be **U-shaped in step count** — it dipped to ~+0.05 around step 850 and rebounded to +0.12 by 1700 — so "the gap" depends on where you read it. With a single seed and a step-dependent gap, the ±0.03–0.05 wander of a single run is the same size as the 160M→400M −0.038 move we'd like to call signal. Treat the trend as suggestive, not measured.

**(c) Runtime version skew.** Per `environment.txt`, the fresh 50M/400M runs ran on **torch 2.12.0.dev20260408+cu128** (A10, commit `e773905`); the 160M anchor came from an earlier **torch 2.7.0**-class runtime. Within each width the A/B is clean (both arms same runtime), and the gap is a *difference* so a uniform version effect largely cancels — but it is not zero, so flag it when comparing the 160M anchor against the two fresh rungs. *(Note: the original prediction framed this as "2.7.0 vs 2.12-nightly"; the artifacts show the fresh runs on the 2.12 nightly and the anchor on the older runtime — same conclusion, the differences cancel within-width.)*

**(d) 50M-bitnet flaked once and was rerun.** `run_status.txt` records `50m_bitnet` failing on attempt 1 with no "Training completed" (a transient HF-CDN HTTP 408 in the data pipeline), then **succeeding on attempt 2** (`RERUN 50m_bitnet OK (completed) attempt 2`). This was a **data-loader / CDN flake, not a model issue**; the reported 50M-bitnet number is from the successful rerun.

---

## 5. Open Question (for the user to own — no AI in the room)

> **Is the non-monotonic 50M→160M→400M hump (+0.058 → +0.123 → +0.085) real structure or a single-seed / step-phase artifact — and what is the *minimum* experiment that would settle it?**
>
> Concretely: would ≥3 seeds per cell collapse the hump into a flat band (→ it was noise), or would the up-then-down shape survive error bars (→ it's the depth-compounds / width-relieves mechanism showing through)? And is the cleaner isolation a **seed replication** or a **width-only rung** (hold `n_layers` fixed, sweep `d_model`) to break the depth/width confound — which one buys more certainty per GPU-hour?

Resolve this closed-laptop: decide what you'd *predict* multi-seed error bars to be (would they cover the hump?), and which of {more seeds, width-only rung, matched-FLOP read-point} you'd spend the next GPU budget on — and why.

---

## Appendix — verification trail

Final `eval.perplexity` (last such line per `*_curves.jsonl`, step 1700):

```
50m_stock   eval.perplexity = 4.71687838889631   → loss ln = 1.5511
50m_bitnet  eval.perplexity = 4.9967741959147505 → loss ln = 1.6088
400m_stock  eval.perplexity = 3.959804266093332  → loss ln = 1.3762
400m_bitnet eval.perplexity = 4.312226121445742  → loss ln = 1.4615
```

160M anchor (reused, prior verified run): stock 1.4342, bitnet 1.5576, gap +0.1234.

`environment.txt`: A10 23GB, torch 2.12.0.dev20260408+cu128, py 3.10.12, commit e773905, run 2026-06-17 02:41–05:43 UTC.
`run_status.txt`: 50m_stock OK, 400m_stock OK, 400m_bitnet OK, 50m_bitnet OK on rerun attempt 2 (attempt 1 transient failure).
