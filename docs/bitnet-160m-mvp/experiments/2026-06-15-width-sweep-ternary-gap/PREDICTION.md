# PREDICTION — BitNet ternary-vs-FP16 gap across model width

**Status: written BEFORE any training run. Committed first by rule ("predictions before experiments — the intuition is the product, not the number").**

Date authored: 2026-06-15
Experiment dir: `docs/bitnet-160m-mvp/experiments/2026-06-15-width-sweep-ternary-gap/`

---

## The question

A/B test BitNet b1.58 **ternary** quantization vs an **FP16** baseline across 3 Llama-style width rungs, on byte-identical data/steps/seed (c4 streaming, 1700 steps, seed 42; only the `BitLinearConverter` differs between arms). The metric is **validation loss in nats**, and the quantity of interest is the **gap = val(bitnet) − val(stock)**.

**Does the ternary-vs-FP16 gap CLOSE as the model grows from 50M → 160M → 400M?**

### Rungs (and the confound, stated up front)

| Rung | d_model | n_layers | ~body params | configs |
|------|---------|----------|--------------|---------|
| 50M  | 768  | 8  | ~50M  | `llama3_50m` / `llama3_50m_bitnet` |
| 160M | 1024 | 14 | ~160M | `llama3_160m` / `llama3_160m_bitnet` |
| 400M | 1536 | 16 | ~400M | `llama3_400m` / `llama3_400m_bitnet` |

**Confound (honest flag):** these rungs co-vary **depth with width**. Going up a rung increases `d_model` *and* `n_layers` at once, so any gap-vs-rung trend is really gap-vs-(width×depth), not gap-vs-width in isolation. Worse, depth and width scale *unevenly*: 50M→160M adds a lot of depth (8→14) and modest width (768→1024), while 160M→400M adds a lot of width (1024→1536) but barely any depth (14→16). So the two steps in this sweep are not even the same *kind* of scaling step. Read every conclusion below as "per existing-rung," not "per parameter," and certainly not "per d_model holding depth fixed."

---

## Hypothesis (one paragraph)

The **dramatic** part of the closure has already happened *below* this sweep's smallest rung. Between the 3.2M char-GPT toy (gap **+1.27 nats**) and 160M (gap **+0.123 nats**) the gap collapsed ~10× over ~1.7 orders of magnitude — that is the real "closes with scale" signal, and it lives mostly under 50M. Inside the narrow 50M→400M window we are sitting on the **shallow tail** of that curve, **~1–2 orders of magnitude below the paper's ~3B crossover**, where the gap is small (order ~0.1 nat) and the remaining motion is comparable to our measurement noise. I therefore predict the gap-vs-width curve over these three rungs is **flat-to-shallow-down — essentially a plateau with a slight downward tilt at the top end — NOT a clean monotonic descent to zero.** I do **not** predict a width at which the gap "starts closing" inside this range, because closure is not a switch that flips here; if forced to name one, the gap is already in its slow-decay regime by 50M. My central number for 400M is a gap **modestly below** 160M (~+0.09, i.e. ~0.03 nats smaller), but I hold low confidence that this 0.03 will exceed the noise floor.

---

## Numeric prediction table

Anchor = the **verified** 160M point (stock 1.4342, bitnet 1.5576, gap **+0.123**, seed 42, 1700 steps — reused, not re-run). Reasoning outward from there.

| Rung | predicted **stock** val | predicted **bitnet** val | predicted **gap** (nats) | basis |
|------|------------------------|--------------------------|--------------------------|-------|
| 50M  | ~1.56 | ~1.61 | **+0.05** | rescued single-copy log (1.5591 / 1.6092); weak corroboration |
| 160M | **1.4342** | **1.5576** | **+0.123** | VERIFIED anchor (reused) |
| 400M | ~1.35 | ~1.44 | **+0.09** | extrapolated; central guess, low confidence |

Predicted gap headline: **50M +0.05 · 160M +0.123 · 400M +0.09.**

**Do I expect 400M's gap < 160M's gap?** Central answer: **yes, but only slightly — by ~0.03 nats** (+0.123 → ~+0.09). I deliberately predict 400M *below* 160M rather than continuing upward, because 160M→400M is a width-heavy step and width is where ternary's representational deficit should be most relieved (more channels → more headroom to absorb the 1.58-bit constraint per weight). But ~0.03 nats is **inside** the band over which I've already watched a single run's gap wander (see noise note), so I am not confident this will be a *visible* effect.

**Stock-loss extrapolation note:** 50M→160M stock val fell 1.5591→1.4342 (−0.125). 160M→400M is 2.5× params but adds little depth, so I expect a smaller absolute drop, ~−0.08 to −0.10 → stock ~1.35. Bitnet tracks ~+0.09 above that → ~1.44.

---

## Reasoning (written down)

### (a) Grounding in the prior data points

The four points we actually have, plotted as gap vs scale:

| scale | gap (nats) | quality of evidence |
|-------|-----------|---------------------|
| 3.2M (char-GPT toy) | **+1.27** | own A/B, but tiny + different arch/data |
| 50M | **+0.05** | rescued single log, uncertain — weak |
| 160M | **+0.123** | VERIFIED, seed-locked, reused as anchor |
| ~3B (paper Table 1) | **~0** | external claim |

The macro story (3.2M → 160M → 3B) is a clean monotonic descent toward zero — that **supports** "closes with scale." But the *micro* story inside our sweep is **not** monotonic: the weak 50M point (+0.05) sits **below** the solid 160M point (+0.123). Taken literally, the gap went *up* from 50M to 160M. I do not trust that bump as physics, for three reasons:

1. **The 50M point is weak** — single copy, uncertain step count, rescued from a terminated box, no seed replication. Treat it as a hint, not a measurement.
2. **The 160M gap is U-shaped in *step* count** — it dipped to **+0.05 at step 850** and rebounded to **+0.12 by step 1700**. So "the gap" is not one number; it depends where in training you read it. If the 50M log was read at a different effective point on its own U, the 50M-vs-160M comparison is confounded by *training progress*, not just scale. Notably, 160M's mid-training gap (+0.05) **equals** 50M's final gap — consistent with the two models being at different phases of the same shape.
3. Single seeds everywhere. We have no error bars.

Net: the prior data is consistent with a **gap that is already small (~0.05–0.13) and roughly flat across 50M–400M**, riding on top of step-phase and seed noise. It is *not* clean enough to assert a downhill slope within the window.

### (b) The paper's scale-crossover claim

BitNet b1.58 reports ternary **matching** FP16 only around **~3B+** params (Table 1); below that ternary underperforms. We are running **50M–400M — roughly 1–2 orders of magnitude below the crossover.** Honest implication: **we should not expect to *see* closure complete in this range.** At best we are measuring the slope of the approach far out on the tail. The expected per-rung change in gap, this far below crossover, is small relative to our noise. So "the gap visibly closes across 50M→400M" is, a priori, an *unlikely* outcome — the more probable honest outcome is "small gap, shallow/ambiguous trend."

### (c) The depth/width confound

Because each rung bumps depth and width together — and unevenly (50M→160M is depth-heavy, 160M→400M is width-heavy) — even a clean monotonic gap-vs-rung result would **not** cleanly isolate "width relieves ternary." A skeptic's read: if ternary's deficit is mostly a *per-layer* error that compounds with depth, then *adding depth could widen the gap* while *adding width narrows it* — and our two steps mix these in opposite proportions. That mechanism alone could manufacture the apparent 50M(+0.05)→160M(+0.123) rise (depth-heavy step → gap up) followed by a 160M→400M fall (width-heavy step → gap down), which is **exactly the central prediction above**. I flag this so we don't over-interpret a non-monotonic curve as either confirming or denying scale-closure: the rungs are not a clean width axis.

---

## Falsifiable bar

Let `g50, g160(=+0.123 fixed), g400` be the measured val-loss gaps. **Noise band: ±0.03–0.05 nats**, justified by the +0.05↔+0.13 wander of a *single* 160M run's gap across the back half of training and the absence of seed replication.

**CONFIRMS the paper's "closes with scale" direction** if:
- `g400 ≤ ~+0.08` (i.e. at least ~0.04 below the 160M anchor, beyond the noise floor), **and ideally**
- the trend is monotone-ish down across rungs once the 50M point is taken at face value (`g50 ≥ g160 ≥ g400` would be the strong form — though note g50<g160 in priors, so the achievable confirm is mainly the 160M→400M leg dropping clearly).

**CONTRADICTS "closes with scale" (at this scale)** if:
- `g400 ≥ g160` (flat or *rising*), e.g. `g400 ≥ +0.13` — the gap does not shrink, or widens, as we scale up. This would say: at 50M–400M, ternary's deficit is **not** relieved by scale (consistent with "all the closure is below 50M and/or beyond 400M, with depth-compounding pushing the wrong way in between").

**WITHIN NOISE / INCONCLUSIVE** if:
- `g400` lands in ~`+0.09` to `+0.15` (within ±0.03 of the anchor). This is the **most likely** outcome by my own estimate: a shallow, ambiguous trend we cannot cleanly call closure on with single seeds.

---

## Intellectual-honesty statement

It is a **legitimate and arguably likely** prediction that the gap does **NOT** meaningfully close across 50M→400M, and that we see — at best — a shallow downward tilt on the 160M→400M leg that may not clear the noise floor. The strong "closes with scale" signal in our own data lives *between 3.2M and 160M*, below this sweep's floor. This experiment's honest job is therefore **not** to "prove closure" but to **measure where on the tail we sit and how steep (if at all) the local slope is**, while being explicit that (i) single seeds give us no error bars, (ii) the gap is U-shaped in training step so the read-point matters, and (iii) depth/width are confounded and scale unevenly between the two steps. If the result is "inconclusive," that is a *correct* result, not a failed one.

### What would upgrade this from a guess to a claim (out of scope here, noted for later)
- ≥3 seeds per cell for real error bars.
- A width-only rung (hold `n_layers` fixed, vary `d_model`) to break the depth/width confound.
- Reading the gap at a **fixed, matched** point on each run's loss curve, not just step 1700, given the U-shape.
