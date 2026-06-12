# 2026-06-11 — Fingerprint smoke (clean-clone, ~20 min budget)

## Goal

A short, mechanism-rich smoke on a fresh Lambda A10 that produces **explicit evidence** the BitNet path is real, rather than another loss curve:

1. **Clean-clone validation of the committed FSDP guard** — doc 12's carry-forward caveat: the degree-1 FSDP skip in `torchtitan/models/llama3/parallelize.py` was committed after the 1700-step run but never re-tested from a fresh clone. This run is that test.
2. **Conversion fingerprint** — reconcile the open `Swapped 84` vs `bitlinear_count 98` mismatch with per-FQN evidence (`fingerprint.py`).
3. **Quantization/STE fingerprints** — ternary weight histogram, int8 activation levels, nonzero grads on forward-clipped latent weights.
4. **Checkpoint-resume probe** — previously listed as unproven: run 100 steps (checkpoints at 50/100), then re-launch with `--training.steps 120` and confirm TorchTitan resumes from step-100.

## Cost guardrail

One Lambda `gpu_1x_a10` (~$1.29/hr), terminated as soon as artifacts are pulled. Target ≤ ~30 min of box time.

## Predictions (written before the run)

1. The BitNet 160M path **runs end-to-end from a clean clone** with no on-box patches — the committed FSDP guard suffices on a single A10.
2. Fake-quant BitNet is **slower than stock** (1700-step run: 1.77×) and **not memory-saving** (slightly higher peak memory), because latent FP weights stay resident and quant temporaries are extra work, not packed kernels.
3. All losses **finite**; 100-step BitNet train loss decreases from ~7–8 toward ~3–4 at seq-len 512 (more tokens/step than the old seq-128 shakedown, but same general shape).
4. Conversion fingerprint: **84 configs converted, 98 BitLinear modules**, explained by `QKVLinear` building `wk` and `wv` from the same `wkv` config (6 configs but 7 modules per layer × 14 layers; code-confirmed pre-run, needs empirical confirmation).
5. Weight fingerprint: exactly ≤ 3 unique forward weight values $\{-\gamma, 0, +\gamma\}$; zero-fraction roughly 30–50% for kaiming-style init under absmean rounding (low confidence on the exact fraction).
6. STE fingerprint: grads nonzero on ~100% of a sample BitLinear's latent weights **including** those clipped to ternary endpoints in the forward.
7. Resume probe: re-launch with `--training.steps 120` loads step-100 and trains 20 more steps; the step-101 loss should be near the step-100 loss (no re-warmup cliff).

## Method

Fresh clone of `carlwilsn/torchtitan@main` on the box, venv with stable cu128 torch (fallback: cu128 nightly if import symbols are missing), then:

```bash
# 0. unit tests
python -m pytest -q tests/unit_tests/test_bitnet_quantization.py

# 1. fingerprint probe (writes fingerprint_report.json)
python docs/bitnet-160m-mvp/experiments/2026-06-11-fingerprint-smoke/fingerprint.py

# 2. 100-step BitNet train w/ checkpoints + validation (config defaults: bs 2, seq 512, ckpt @50, val @50)
MODULE=llama3 CONFIG=llama3_160m_bitnet NGPU=1 LOG_RANK=0 ./run_train.sh \
  --parallelism.data-parallel-shard-degree 1 \
  --checkpoint.enable \
  --job.dump-folder ./outputs/fingerprint_smoke

# 3. resume probe: same command, --training.steps 120 → must load step-100
MODULE=llama3 CONFIG=llama3_160m_bitnet NGPU=1 LOG_RANK=0 ./run_train.sh \
  --parallelism.data-parallel-shard-degree 1 \
  --checkpoint.enable \
  --training.steps 120 \
  --job.dump-folder ./outputs/fingerprint_smoke
```

## Results

Status: **completed 2026-06-12 UTC** on a fresh Lambda `gpu_1x_a10` (NVIDIA A10 23028 MiB, driver 570.148.08), clean `--depth 1` clone at commit `3856f814a`, venv torch `2.12.0.dev20260408+cu128` (the exact nightly of the 1700-step run). Box terminated after artifact pull.

### Environment detour (documented, not hidden)

Stable `torch 2.11.0+cu128` was tried first and **failed at import**: `cannot import name 'DataParallelMeshDims' from 'torch.distributed.fsdp'`. This checkout has moved past 2.11 — the cu128 **nightly** is currently required. (Unit tests don't touch `parallelize.py`, so they passed even on 2.11 — they are not an env gate for the trainer.)

Also: the dump-folder flag for this config schema is `--dump-folder`, **not** `--job.dump-folder` (tyro rejects the latter with exit code 2 and an "Unrecognized options" panel that only shows with `--tee`-captured rank logs).

### Prediction scorecard

| # | Prediction | Outcome |
| --- | --- | --- |
| 1 | Clean clone runs with committed FSDP guard, no on-box patches | ✅ **Confirmed.** Zero patches applied on the box. 100-step + resume runs completed; backward worked. Doc 12's carry-forward caveat is retired. |
| 2 | Fake-quant slower than stock, not memory-saving | ☑️ Not re-measured (no stock run this time); stands on the 1700-step evidence (1.77× slower, +1.47 GiB). |
| 3 | Losses finite, ~7–8 → ~3–4 by step 100 | ✅ Step 1: `7.98490` → step 100: `2.66239`, all finite (slightly better than predicted band). |
| 4 | 84 configs vs 98 modules, explained by shared `wkv` config | ✅ **Confirmed empirically**: `configs_converted: 84`, `bitlinear_modules_built: 98`, FQN lists show `…qkv_linear.wkv` (1 config) → `…qkv_linear.wk` + `…wv` (2 modules); exactly 7 BitLinear per layer × 14 layers. **Mismatch reconciled — it's correct behavior, not a bug.** |
| 5 | ≤3 unique forward weight values, zero-fraction 30–50% | ⚠️ **Almost — instructive miss.** Values collapse to $\{-\gamma, 0, +\gamma\}$ ($\gamma=0.01598$) at 1e-6 precision, but `torch.unique` finds **7 bit-distinct values**: the STE form `w + (y - w).detach()` reconstructs `y` with ~1-ulp float error, so forward weights are ternary only up to rounding. Zero-fraction `0.310` (in predicted band). |
| 6 | Grads nonzero incl. forward-clipped weights | ✅ 100% nonzero grad fraction on the sample weight; 242,978 weights (23%) were clipped to ternary endpoints in forward and **all** received gradient — identity STE active. |
| 7 | Resume loads step-100, no re-warmup cliff | ✅ `Loading the checkpoint from …/step-100` (1.11 s), step 101 loss `2.81932` vs step 100 `2.66239` (within step-to-step noise band ~2.6–3.0), ran to 120, step-120 checkpoint saved. |

### Key numbers (BitNet `llama3_160m_bitnet`, bs 2, seq 512, A10)

| Quantity | Value |
| --- | --- |
| Converter log | `Swapped 84 Linear layers to BitLinear` |
| Params | 160,062,976 |
| Step 1 / 50 / 100 loss | 7.98490 / 3.39110 / 2.66239 |
| Resume: step 101 / 120 loss | 2.81932 / 2.85589 |
| Throughput | ~3,360 tps steady (seq 512 vs old shakedown's seq 128) |
| Peak logged memory | 1.33–1.35 GiB (6% of A10) |
| Checkpoints | step-50, step-100, step-120 (`artifacts/checkpoint_files.txt`) |
| `lm_head` / `tok_embeddings` | `Linear` / `Embedding` (unquantized, as designed) |
| Stock-vs-BitNet forward (same seed/tokens) | mean abs logit diff 1.11, max 6.58 → quantization demonstrably in the forward path; stock model contains 0 BitLinear |
| Unit tests (GPU, clean clone) | `4 passed` |

### Honest gaps

- **Validation never emitted** in the 100-step run despite `validator.freq=50, steps=10` config defaults — same gap as the 2026-06-08 shakedown. The 1700-step run *did* get eval losses with explicit `--validator.freq/--validator.steps` CLI flags; whether config-default validator settings are honored without CLI flags is still unconfirmed. Worth a 5-minute local probe before relying on defaults.
- Prediction 2 (speed/memory vs stock) was not re-measured here — no stock training run in this smoke.
- Single GPU only; FSDP multi-rank path still unexercised.

### Artifacts

- `fingerprint_report.json` — full probe output (conversion FQNs, ternary histogram, STE grads, logit diffs).
- `artifacts/environment.txt` — GPU/torch/commit provenance.
- `artifacts/train100.log`, `artifacts/resume120.log` — raw trainer logs; `*_steps.txt` are ANSI-stripped per-step lines.
- `artifacts/smoke2.log` — unit-test + fingerprint console output (clean-clone hash visible).
- `artifacts/checkpoint_files.txt` — checkpoint listing (payloads not kept).

### What this retires / opens

Retired: doc 12's FSDP-guard clean-clone caveat; the 84-vs-98 converter mystery (shared `wkv` config → wk+wv modules); checkpoint-resume "unproven" status.

Opened: forward weights are *almost*-ternary (ulp-level error from the STE residual form) — harmless for training, but worth owning **why** `w + (y-w).detach()` is used instead of returning `y` with a custom autograd function (gradient routing, not numerics). Good closed-laptop derivation candidate.
