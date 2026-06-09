# 12 — The 1700-step seed-locked stock-vs-BitNet run, read like a scientist

This is the analysis of the **2026-06-09 1700-step seed-locked run** — the first comparison that clears the bar doc 11 set: a real workload (seq-len 2048, batch 16), seed-locked init, validation that actually emits a loss/perplexity, checkpoints, and a token budget four orders of magnitude larger than the shakedown. Doc 11 proved *the loop works*. This doc asks: *with the defects fixed and a real budget spent, what can we now honestly conclude about the FP-vs-ternary gap?*

Raw data (the evidence behind every number here):

- `../results/stock_curves.jsonl` — filtered `train.loss` / `eval.loss` / `eval.perplexity` events, stock `llama3_160m`.
- `../results/bitnet_curves.jsonl` — same, `llama3_160m_bitnet`.
- `../results/results_summary.txt` — the extracted final numbers + curves.

## Run shape (byte-identical except the converter)

Both runs used **identical** training flags; only `CONFIG` (`llama3_160m` vs `llama3_160m_bitnet`) and the dump-folder differed. That fairness is the whole experiment.

| Setting | Value |
| --- | --- |
| Steps | 1700 |
| Local batch size | 16 |
| Sequence length | 2048 |
| Tokens / step | $16 \times 2048 = 32{,}768$ |
| Total tokens / run | $1700 \times 32{,}768 \approx 55.7\text{M}$ |
| Dataset | `allenai/c4` (real streaming, **not** the tiny `c4_test` fixture) |
| Activation checkpoint | `selective` |
| Seed | `42` (both, via the committed `debug.seed=42` fix) |
| Validation | `--validator.freq 170 --validator.steps 20` (every ~10%) |
| Checkpoint | every 850 steps, keep-latest-2 |
| Hardware | single Lambda A10 (24 GB), torch `2.12.0.dev20260408+cu128` |

This is ~`55.7M` tokens per run versus the shakedown's `12,800` — about **4,350× more tokens**. The shakedown's central limitation (loss noise an order of magnitude larger than the gap) is exactly what this budget was sized to beat.

> **Env note (carry forward):** the degree-1 single-GPU FSDP guard skips `apply_fsdp` when no parallel degree actually shards/replicates the model. Without it, `loss.backward()` crashes with "tensor … data is not allocated yet" (the device mesh logs `active dimensions: []`, confirming FSDP was correctly skipped and training ran dense on one GPU). At run time this was an **uncommitted on-box patch**; it is **now committed** in `torchtitan/models/llama3/parallelize.py` — it early-returns when `fsdp_enabled`/`dp_replicate_enabled`/`pp_enabled`/`full_dtensor` are all false, using the framework's own `parallel_dims.fsdp_enabled` property (= `dp_shard>1 or cp>1`) so any real multi-GPU config runs the original FSDP path unchanged. **Honest caveat:** the committed form mirrors the on-box logic that ran these 1700 steps but was not itself re-smoke-tested on a fresh box — the first clean-clone run should confirm `loss.backward()` succeeds before trusting it blindly.

---

## 1. The headline numbers

Straight from `results_summary.txt`:

| Quantity | Stock | BitNet | Gap (BitNet − stock) |
| --- | ---: | ---: | ---: |
| Final **train** loss (step 1700) | 1.4123 | 1.5438 | **+0.1314 nats** |
| Final **val** loss (step 1700) | 1.4342 | 1.5576 | **+0.1233 nats** |
| Final **val perplexity** | 4.1965 | 4.7473 | +0.55 ppl |
| Median throughput (tps) | 33,111 | 18,697 | **1.77× slower** |
| Peak GPU memory | 8.34 GiB | 9.81 GiB | **+1.47 GiB** |
| grad_norm (tail) | ~0.5–0.7 | ~1.4–1.6 | BitNet noisier |

The two big quality numbers — **train gap +0.131, val gap +0.123** — agree closely, which is the first reassuring sign: the gap is not an artifact of the training set, it shows up identically on held-out validation.

> **Caveat that keeps this honest:** this is **one seed**. The goal-file plan calls for 3 seeds before any of this is a *quality claim*. What we have is a single, clean, internally-consistent data point — strong enough to compare against predictions, not yet strong enough to publish a "BitNet costs X nats" headline.

---

## 2. The eval-loss curve — and the U-shaped gap

The validation loss at every `validator.freq` boundary, both runs (from `results_summary.txt`):

| Eval step | Stock | BitNet | Gap |
| ---: | ---: | ---: | ---: |
| 1 | 7.1211 | 5.8622 | −1.2589 |
| 170 | 2.3265 | 2.5295 | +0.2030 |
| 340 | 1.9896 | 2.1447 | +0.1551 |
| 510 | 1.8081 | 1.8862 | +0.0781 |
| 680 | 1.7315 | 1.7926 | +0.0611 |
| 850 | 1.6844 | 1.7356 | +0.0512 |
| 1020 | 1.6245 | 1.6945 | +0.0700 |
| 1190 | 1.5485 | 1.6567 | +0.1082 |
| 1360 | 1.4848 | 1.6082 | +0.1234 |
| 1530 | 1.4548 | 1.5805 | +0.1257 |
| 1700 | 1.4342 | 1.5576 | +0.1233 |

Two things matter here.

**2.1 The step-1 "gap" is meaningless (and negative).** At step 1, BitNet's eval loss (5.86) is *lower* than stock's (7.12). This is the same trap doc 11 §1.4 warned about: from identical seed-42 init weights, BitNet's quantized forward produces a flatter, re-centered output distribution on the very first batch, which happens to score lower CE before any learning. It says nothing about quality. **Do not quote step-1.**

**2.2 The gap is U-shaped, not monotone.** It opens at +0.20 (step 170), **narrows to a minimum of +0.051 at step 850**, then **widens back to +0.123 by step 1700**. This is the single most interesting curve in the run, and the mechanism is worth owning:

- *Early (steps 0–850):* both models dump the easy entropy (unigram/bigram structure). Ternary weights are plenty expressive for the coarse statistics, so the gap shrinks — the quantized model keeps pace on the easy gains.
- *Late (steps 850–1700):* stock starts exploiting fine-grained weight precision to fit subtler structure that ternary weights literally cannot represent. The gap reopens and stabilizes around +0.12.

This is the **expected signature of a representational-capacity gap**, not a bug. A bug would look like a gap that grows *without bound* or a loss that trends *up*. Here the gap plateaus at ~+0.12 and both losses keep falling. (See §4, watch-signal #3.)

> **Load-bearing intuition (re-derivable):** a ternary weight can take only 3 values per element (scaled by one per-tensor $\alpha$). Early training only needs to get the coarse direction of each weight right — ternary does that fine. Late training is about fine adjustments to many weights at once; that is precisely the precision a 3-level weight throws away. So a gap that *shrinks then reopens* is what capacity-limited quantization should look like at this scale.

---

## 3. The throughput and memory cost — confirming doc 11's mechanism

| Metric | Stock | BitNet | Ratio / delta |
| --- | ---: | ---: | ---: |
| Median tps | 33,111 | 18,697 | **1.77× slower** |
| Peak memory | 8.34 GiB | 9.81 GiB | **+1.47 GiB** |

The **1.77× slowdown** is right in the doc-11 ballpark (1.90× at the tiny shape) and confirms the same mechanism: this MVP is **fake-quant in FP**. Every `BitLinear` forward does RMSNorm + activation quant + weight ternarization + **the same dense `F.linear` stock would have done** — strictly more FP work, so it must be slower. The ratio being a touch lower than 1.90× here is plausibly because at batch-16/seq-2048 the dense matmul is a larger fraction of the step, so the fixed quant-pass overhead amortizes slightly better.

The **+1.47 GiB** memory is much larger in absolute terms than the shakedown's +0.09 GiB — but that's just scale (batch 16 vs 1, seq 2048 vs 128 → far bigger activation tensors, and BitNet's extra quant/RMSNorm temporaries scale with them). The *sign* is the same and for the same reason: FP master weights stay resident, and the quantized forward allocates extra same-size temporaries. The real 1.58-bit memory win lives in **post-packing inference**, which this run does not exercise.

---

## 4. Predictions vs. results (stated before the run, in the goal file)

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 1 | Stock final train loss ~3.0–3.6 nats | **1.41** | **Miss (better than predicted)** — c4 + a real batch/seq + 55.7M tokens drove loss well below the conservative band. The prediction assumed a weaker workload. |
| 2 | BitNet train loss above stock by ~0.1–0.4 nats | **+0.131** | **Hit** — squarely in the predicted band, low end. |
| 3 | Gap stable or slowly shrinking, NOT persistently growing | U-shaped: shrinks to +0.051 (step 850), reopens to +0.123, then **plateaus** | **Hit (with nuance)** — not monotone-shrinking, but it stabilizes (no blow-up, no unbounded growth). The reopening is capacity-limited quantization, not a bug. |
| 4 | BitNet ~1.9× slower, memory +~0.09 GiB | **1.77× slower, +1.47 GiB** | **Partial** — slowdown close (1.77 vs 1.9). Memory delta far larger in absolute GiB, but that's a scale effect (16×bigger batch/seq); same sign, same mechanism. The +0.09 GiB figure was specific to the tiny shakedown shape. |
| 5 | grad_norm both stable, BitNet slightly noisier | Stock tail ~0.5–0.7, BitNet ~1.4–1.6, no blow-ups | **Hit** — BitNet measurably noisier, both bounded. |
| 6 | `eval.loss`/`eval.perplexity` emitted for BOTH every freq | Emitted at steps 1/170/.../1700 for both (the whole point of Fix 1) | **Hit** — the validation-logging fix works end to end. |
| 7 | No memory leak; checkpoint write/resume works | Memory flat at 8.34/9.81 GiB across all 1700 steps; checkpoints at step-850 + step-1700 for both, keep-latest-2 honored | **Hit (write); resume not separately exercised)** — no leak, writes confirmed. A cold resume-from-checkpoint was not run as a separate step here. |

**Scorecard: 4 clean hits, 1 hit-with-nuance (#3), 1 partial (#4), 1 miss (#1, in the good direction).**

The one outright miss (#1) is the most instructive: the prediction under-rated the workload. Switching from the tiny `c4_test` fixture to real streaming `c4` with a proper batch/seq and 55.7M tokens let *both* models reach ~1.4–1.5 nats, far below the "won't converge" band. That's a good miss — it means the run was a real training run, not a toy.

---

## 5. The seven watch-signals (from doc 11 §4)

| # | Signal | Observed | Verdict |
| --- | --- | --- | --- |
| 1 | Non-mirrored loss spikes | None. Both descend smoothly; no BitNet-only spikes. | ✅ clean |
| 2 | grad_norm blow-ups | Bounded throughout (stock ~0.5–0.7, BitNet ~1.4–1.6 tail). | ✅ clean |
| 3 | Persistently **growing** gap | Gap is U-shaped and **plateaus** at ~+0.12; does not grow without bound. | ✅ capacity gap, not bug |
| 4 | Throughput drift | tps flat near medians (stock ~33k, BitNet ~18.7k) the whole run. | ✅ no drift |
| 5 | Memory growth / leak | Flat at 8.34 / 9.81 GiB for all 1700 steps. | ✅ no leak |
| 6 | Checkpoint write + resume | Writes confirmed (step-850, step-1700, both runs). Resume not separately tested. | ✅ write / ⚠ resume untested |
| 7 | Validation emits a real loss/ppl | `eval.loss` + `eval.perplexity` for both at every freq. | ✅ the Fix-1 payoff |

Six of seven fully clean; the seventh (checkpoint **resume**) had its write half proven and its load half deferred — that's the one concrete follow-up this run leaves open.

---

## 6. What this run does and does NOT license

**Does:**

- The validation-logging fix (`eval.loss`/`eval.perplexity` to JSONL) works on a real run.
- Seed-lock works: step-1 train loss was bit-identical to the throughput probe (8.25559), confirming deterministic init + data order.
- At 160M / 55.7M tokens on c4, the FP-vs-ternary **train and val gaps agree** (+0.131 / +0.123) — the gap is real and held-out, not a training-set artifact.
- The fake-quant cost (1.77× slower, more memory) is confirmed at a real workload scale, same mechanism as the shakedown.

**Does NOT (yet):**

- **A multi-seed quality claim.** This is one seed. The plan needs ≥3 (`--debug.seed N`) before "+0.12 nats" is a number you defend. The gap could move ±some seed-spread we haven't measured.
- **Convergence.** Both losses were still falling at step 1700. We measured a *snapshot* of the gap at 55.7M tokens, not an asymptotic gap.
- **Checkpoint resume correctness** for BitNet (load + continue without key mismatch).
- **Multi-GPU / real FSDP.** This ran with FSDP skipped at degree-1. Sharded behavior is untested.
- **Any packed-ternary inference benefit.** This is the training-time fake-quant end of the tradeoff.

---

## 7. One-paragraph honest summary

The 1700-step seed-locked run is the first comparison that clears doc 11's bar. With identical flags, seed-42 init, real streaming c4, and ~55.7M tokens per run, stock `llama3_160m` reached train/val loss **1.41 / 1.43** (ppl 4.20) and BitNet reached **1.54 / 1.56** (ppl 4.75) — a **+0.13-nat gap that agrees on train and held-out validation**, well below the shakedown's noise floor and now a real (if single-seed) signal. The gap is **U-shaped**: it shrinks to +0.05 by step 850 then reopens to +0.12 and plateaus — the expected signature of a representational-capacity limit in ternary weights, not a bug (no unmirrored spikes, no grad blow-ups, no unbounded growth, no leak). The fake-quant cost held at **1.77× slower / +1.47 GiB**, same mechanism as the shakedown at larger scale. Of seven predictions: four clean hits, one nuanced hit, one partial, one good-direction miss (stock converged far better than the conservative band because the workload was real). Six of seven watch-signals fully clean; checkpoint **resume** is the one deferred follow-up. The remaining gates before this is a publishable quality claim are **multiple seeds**, **a resume test**, and eventually **multi-GPU FSDP** — none of which this single A10 run was scoped to cover.
