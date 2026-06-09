# 11 — Reading the 100-step shakedown like a scientist

This page is the **analysis** of the 2026-06-08 100-step stock-vs-BitNet shakedown, not a re-summary of it. The experiment README answers "what did we run and what were the headline numbers." This page answers a harder question: *given these exact CSVs, what can we honestly conclude, what is just noise, and what should we instrument before the next run?*

Source data (read end to end for this page):

- `../experiments/2026-06-08-100-step-shakedown/artifacts/stock_loss_metrics.csv`
- `../experiments/2026-06-08-100-step-shakedown/artifacts/bitnet_loss_metrics.csv`
- `../experiments/2026-06-08-100-step-shakedown/artifacts/parsed_summary.json`
- `../experiments/2026-06-08-100-step-shakedown/artifacts/stock_structured_logs/*.jsonl`
- `../experiments/2026-06-08-100-step-shakedown/artifacts/bitnet_structured_logs/*.jsonl`

Run shape (both runs identical except the BitNet converter): `100` steps, `seq_len=128`, `local_batch_size=1`, single A10, no grad accumulation. So each optimizer step sees exactly **128 tokens**, and the whole run touches roughly $100 \times 128 = 12{,}800$ tokens. That number is the lens for everything below: it is about four orders of magnitude short of a quality measurement.

---

## 1. What the loss curves actually do

### 1.1 The numbers, taken straight from the CSVs

| Quantity | Stock | BitNet | Notes |
| --- | ---: | ---: | --- |
| Step-1 loss | 8.12197 | 7.99145 | $\ln(2048)\approx 7.62$ is the uniform-prior reference for a 2048 vocab |
| Step-100 loss | 2.78636 | 2.75976 | last single step, itself noisy |
| Min loss (step) | 2.71447 (89) | 2.64968 (96) | the minima land at different steps |
| Max loss (step) | 8.12197 (1) | 7.99145 (1) | both maxima are step 1 |
| First-10 avg | 7.61875 | 5.67436 | BitNet's first-10 is lower **because of init/data, not quality** (see §1.4) |
| Last-10 avg | 2.86142 | 2.82334 | the only semi-stable comparison, gap `0.038` |
| Last-20 avg | 2.85940 | 2.82540 | gap `0.034` |
| First step < 4.0 | step 20 | step 13 | |
| First step < 3.0 | step 48 | step 48 | identical |

Step 1 starts a hair above $\ln(\text{vocab})=\ln(2048)\approx 7.62$, exactly where a barely-initialized model should sit: it is predicting close to uniform over the 2048-token vocabulary, so cross-entropy $\approx \ln(2048)$. That both runs start near this value is a sanity check that init and the loss wiring are sane, nothing more.

### 1.2 Shape of the descent

Both curves do the same three things:

1. **Steep early drop.** Step 1 → step 20 falls `53.2%` for stock (8.12 → 3.80) and `56.4%` for BitNet (7.99 → 3.48). This is the model dumping the easy entropy — learning unigram frequencies and the most common bigrams. It is not "learning language," it is learning the marginal token distribution.
2. **Flatten around step ~25–48.** Both cross below 3.0 at exactly step 48. After ~step 30 the trend is nearly flat and the step-to-step movement is dominated by which batch happened to land.
3. **Noisy plateau, steps ~50–100.** The tail standard deviation is `0.327` (stock) and `0.323` (bitnet) over steps 50–100. That is the size of the per-step wobble — and it is **much larger than the 0.038 gap between the two models' last-10 averages.**

```text
loss tail (steps 50-100):
  stock   mean ~3.10   pstdev 0.327
  bitnet  mean ~3.06   pstdev 0.323
  stock-bitnet last-10 mean gap = 0.038   <-- ~9x smaller than one run's own wobble
```

### 1.3 The two curves are essentially the same curve

The single sharpest fact on this page. Compute the Pearson correlation between the stock loss series and the BitNet loss series, step-matched:

| Window | corr(stock loss, bitnet loss) |
| --- | ---: |
| Steps 1–100 | 0.907 |
| Steps 20–100 (after the steep drop) | **0.986** |

A correlation of `0.986` over steps 20–100 means the two models go up and down **together, batch by batch.** Look at the visible bumps and you can see why:

| Step | Stock loss (grad_norm) | BitNet loss (grad_norm) |
| --- | --- | --- |
| 29 | 4.019 (6.44) | 3.899 (4.84) |
| 30 | 5.198 (8.44) | 5.116 (7.41) |
| 31 | 6.165 (9.88) | 5.690 (8.13) |
| 57 | 4.373 (7.59) | 4.320 (5.53) |

Both models spike on step 30–31 and step 57. They cannot both have a "bad model moment" on the same steps independently — what they share is the **data order**. Steps 30, 31, 57 are simply hard / high-entropy batches in this fixed data stream, and both models pay for them. The variance you see in the loss is variance in *the data*, not variance in *the model*.

> **Load-bearing intuition (re-derivable):** at batch size 1 and seq-len 128, each loss value is the average NLL over 128 tokens of one specific window of text. With no averaging across a large batch, the per-step loss is a high-variance estimate of the true loss. Two models trained on the *same* stream will have nearly identical per-step loss noise. So a step-100 loss difference of `0.027` is noise riding on a shared signal — not a measurement of either model.

### 1.4 Why BitNet's *early* loss looks lower (and why that's a trap)

BitNet's first-10 average (`5.67`) is much lower than stock's (`7.62`). It is tempting to read this as "BitNet learns faster." It is not. Two mundane causes fully explain it, and neither is about model quality:

- **Different init draw.** Stock and BitNet were *not* seed-locked to produce identical initial weights in this run (matched *settings*, not verified identical RNG state — see §3). Different init → different first few losses.
- **The quantizer reshapes early dynamics.** BitNet's weight quantization re-centers and rescales the effective weights every forward (see §2), which changes the early loss landscape. This shifts the first handful of steps; it says nothing about where the model converges.

By step 48 both models are at the same place (both first cross 3.0 at step 48). The early gap closed. **Do not quote the first-10 averages as evidence of anything.**

### 1.5 Verdict on §1

- Both runs trained: loss fell from ~8 to ~2.8 and stayed down. The training loop, loss, and optimizer are wired correctly.
- The stock-vs-BitNet gap at matched steps (`0.034`–`0.038` on the last-10/last-20 average) is **roughly an order of magnitude smaller than each run's own tail noise (`~0.33`)**. It is inside the noise band, full stop.
- At `12,800` tokens, with init and data order dominating and a `0.986` step-matched correlation, **these loss numbers are not evidence that either model is better.** Anyone who reports "BitNet beat stock by 0.027" from this run has mistaken noise for signal.

---

## 2. The throughput cost is real — and it's an artifact of the naive implementation

### 2.1 The measured cost

| Metric | Stock | BitNet | Ratio |
| --- | ---: | ---: | ---: |
| Mean tokens/sec (steps with tps > 500) | 3643 | 1913 | **1.90× slower** |
| Median tokens/sec | 3687 | 1931 | 1.91× |
| Wall clock (100 steps) | 16.93 s | 20.18 s | 1.19× |
| Max logged GPU memory | 1.28 GiB | 1.37 GiB | +0.09 GiB |

(Step 1 and step 51 tokens/sec are excluded from the throughput means: step 1 includes warmup/compile, and step 51 — `67` tps stock, `60` tps bitnet — is the checkpoint-save stall at the step-50 checkpoint interval. Those are I/O artifacts, not steady-state throughput. The wall-clock ratio is smaller than the tps ratio precisely because the fixed warmup + checkpoint overhead is amortized across both runs.)

### 2.2 The mechanism: this MVP does *strictly more* FP work, not less

This is the part to actually understand. The MVP uses **fake quantization** in ordinary PyTorch ops. It does **not** use packed ternary storage or a ternary matmul kernel. Per `BitLinear` forward it does, in full precision:

1. RMSNorm of the activations,
2. activation quantization (per-token int8-style affine quant → dequant),
3. weight quantization (ternarize the latent FP weight → dequant back to FP),
4. a **normal dense `F.linear` in full precision** on the dequantized operands.

The ternary weight at the math level is

$$
\tilde{W} = \alpha \cdot \mathrm{round\_clip}\!\left(\frac{W}{\alpha},\, -1,\, 1\right),
\qquad
\alpha = \frac{1}{nm}\sum_{i,j} |W_{ij}|,
$$

but $\tilde{W}$ is **materialized as a full FP tensor** and fed into the same `F.linear` stock would have used. So BitNet's forward = stock's forward **plus** RMSNorm + activation quant + weight quant + the round/clip/scale arithmetic, all on FP tensors of the same size. More work in, same matmul, so:

> **It must be slower.** A ~1.9× slowdown is the expected sign and roughly the expected magnitude for "do the FP matmul anyway, then add several elementwise passes over the same tensors." This is not a property of ternary networks — it is a property of *simulating* them in FP.

### 2.3 Why memory is *higher*, not lower

The `+0.09 GiB` looks backwards if you think "1.58-bit weights should be small." It isn't backwards, for the same reason:

- The **FP master ("latent") weights are still resident** — they are the trainable parameters. Quantization happens on the fly in the forward; nothing is stored in packed ternary form.
- The forward **creates extra temporaries**: the dequantized ternary weight tensor $\tilde{W}$, the quantized/dequantized activations, and the RMSNorm intermediates. These are additional same-size FP allocations that the stock path never makes.
- Param count even ticks up slightly (`159,937,536` stock vs `160,062,976` bitnet) because the BitNet blocks carry the extra norm parameters.

The real 1.58-bit memory win lives entirely in **inference after packing** — a packed ternary weight stores ~`1.58` bits/param instead of `16`, an ~`10×` weight-memory reduction. None of that is exercised here. This run measures the *training-time fake-quant* cost, which is the opposite end of the tradeoff.

> **Re-derivable claim:** fake-quant training is *expected* to be slower and slightly heavier than the FP baseline. If a future run shows BitNet *faster* at training time, something changed (a real kernel landed) and you should be suspicious until you can point to that change.

---

## 3. What is just noise at 100 steps (what you cannot conclude)

Enumerated plainly, because the temptation to over-read a finished-looking run is strong:

1. **Relative model quality.** Covered in §1. The gap is inside the noise band; the curves are `0.986` correlated on the data. No quality conclusion is licensed.
2. **Convergence behavior.** 100 steps / `12,800` tokens shows the model dumping easy entropy and then sitting on a noisy plateau. We have not seen it approach any asymptote. "It flattened" here means "the easy gains are gone," not "it converged."
3. **Stability over a real run.** No divergence in 100 steps says nothing about step 5,000. Quantized training can be stable early and destabilize later (or vice versa). The early grad-norm spikes (stock `452` at step 8; bitnet `76` at step 1) settled, but 100 steps is far too short to claim "stable."
4. **Validation / generalization.** This is subtle and important: the structured logs **do** contain `local_valid_tokens` and `global_valid_tokens` events at every step — but those are **counts of non-pad tokens in each training batch (all `128.0`)**, *not* a validation-set loss. There is **no validation-loss event** anywhere in either JSONL (the only `event_name` values present are `local_valid_tokens` and `global_valid_tokens`). So `--validator.freq 50 --validator.steps 2` was configured, but **no validation metric was emitted.** Anyone grepping for "valid" will hit the token-count events and might wrongly conclude validation ran. It did not produce a loss. Treat validation as **unconfirmed**.
5. **Identical-init / true matched comparison.** The runs matched *settings*. We have not verified the two runs shared RNG state / identical initial weights (the divergent first-10 losses in §1.4 are consistent with different init draws). Until seed/init parity is confirmed, even a clean longer run is a *settings-matched* comparison, not a *seed-matched* one.
6. **Checkpoint resume.** Checkpoints **saved** at steps 50 and 100 for both runs (`.metadata` + `__0_0.distcp` present). Save ≠ resume. Loading a BitNet checkpoint and continuing without key mismatches is untested.
7. **Multi-GPU / FSDP correctness.** The single-rank FSDP bypass workaround (`Skipping FSDP for single-rank controlled BitNet experiment`) was still active. Nothing about sharded behavior is tested.
8. **The converter accounting gap.** The `84` logged swaps vs `98` directly counted `BitLinear` modules mismatch (docs 09/10) is unresolved. We are not yet certain *exactly which* layers are ternary.

---

## 4. Watch-list for the real 1–2h run

Each item names the **specific signal** in the CSV / structured logs to watch, and the **trigger** that means "stop and look."

| # | What to watch | Signal (column / event) | Healthy | Stop-and-look trigger |
| --- | --- | --- | --- | --- |
| 1 | Loss divergence / spikes | `loss` column over the whole run | Smooth descent then slow grind; spikes coincide with hard batches in *both* runs | A spike in BitNet **not** mirrored in stock at the same step; or loss trending **up** over a sustained window |
| 2 | Gradient health | `grad_norm` column | Settles to single digits and stays bounded (this run: stock tail mean `4.5`, bitnet `3.2`) | grad_norm climbing across many steps, or sudden `>100` spikes late in training (early spikes like step-8 `452` are expected warmup) |
| 3 | The FP-vs-ternary gap actually opening or closing | step-matched `loss` difference, averaged over a window (not single steps) | Gap stays within a windowed noise band that **shrinks** as the batch count grows (more tokens → tighter loss estimate) | A *persistent, growing* gap in either direction over thousands of steps — that would be the first real quality signal |
| 4 | Throughput stability | `tps` column (ignore step 1 and checkpoint-interval steps) | Flat near the steady-state median (~1900 bitnet / ~3650 stock at this shape; absolute numbers shift with seq-len/batch) | tps drifting **down** over time (thermal throttle, memory fragmentation, leak) |
| 5 | Memory growth | `memory_gib` column | Flat after warmup (this run: pinned at `1.28` stock / `1.37` bitnet for all 100 steps) | Monotonic climb across steps → leak / accumulating temporaries; risk of OOM on the longer run |
| 6 | Checkpoint resume | resume from a saved `step-N` checkpoint; first post-resume `loss` and `grad_norm` | Loss continues from roughly where it stopped; no missing/unexpected state-dict keys for `BitLinear` / pre-norm params | Loss jumps back toward `~8` (lost optimizer/weights), or load errors on BitNet-specific keys |
| 7 | Validation actually emits a metric | a real **validation-loss** `event_name` in the JSONL — **not** `local_valid_tokens`/`global_valid_tokens` | A distinct val-loss event appears every `validator.freq` steps for both runs, at a matched token budget | Only the `*_valid_tokens` count events present → validation still silently not producing a loss (the exact failure mode of this shakedown) |

### 4.1 The single comparison worth making on the long run

Once validation emits a real loss and seed/init parity is confirmed, the comparison to report is **validation loss (or perplexity) at a matched token budget**, with a windowed training-loss curve as support — *not* a single step-100 training-loss number. The decision rule:

$$
\text{quality gap} = L^{\text{val}}_{\text{bitnet}}(T) - L^{\text{val}}_{\text{stock}}(T)
$$

evaluated at the **same number of training tokens** $T$, ideally across multiple seeds. Only when this gap is **larger than the seed-to-seed spread** does it become a quality claim. Everything in this shakedown is below that bar by design.

---

## 5. One-paragraph honest summary

The 100-step shakedown proves the *experiment loop*, not the *model*. Both variants trained (loss ~8 → ~2.8), saved checkpoints at steps 50 and 100, and terminated cleanly. The stock-vs-BitNet loss gap (`~0.035` on the tail average) is roughly `9×` smaller than each run's own tail noise (`~0.33`), the two loss series are `0.986` correlated step-for-step (they ride the same data-order bumps), so **no quality conclusion is licensed.** BitNet ran `~1.9×` slower and used `+0.09 GiB` — exactly as expected for fake-quant-in-FP training, which does the dense FP matmul *plus* quantization passes and keeps FP master weights resident; this is an artifact of the naive implementation, not a property of ternary networks, and is the opposite end of the tradeoff from the real (post-packing, inference-time) `1.58`-bit memory win. Validation was configured but produced **no loss metric** — the JSONL only carries per-step token-count events that are easy to mistake for validation. The next run's job is to graduate from "the loop works" to a seed-matched, validation-backed, multi-thousand-step comparison, watching the seven signals above.
