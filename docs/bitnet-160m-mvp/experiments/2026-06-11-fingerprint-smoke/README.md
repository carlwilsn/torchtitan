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

*(to be filled in after the run)*
