# 08 — Known gaps and next experiments

This page separates what is proven from what remains to be done.

## Proven so far

The following are proven by code/tests/smoke runs:

1. `activation_quant` preserves shape/dtype and passes gradients.
2. `weight_quant` produces scaled ternary forward values and passes gradients.
3. `BitLinear` can run forward and backward in isolation.
4. `BitLinearConverter` can replace matching `Linear.Config` entries in a config tree.
5. TorchTitan can register a 160M Llama-style model flavor.
6. TorchTitan can register stock and BitNet 160M training configs.
7. On Lambda A10, stock debug training can execute 3 steps with the smoke workaround.
8. On Lambda A10, stock 160M training can execute 3 steps with the smoke workaround.
9. On Lambda A10, BitNet 160M training can execute 3 steps with `BitLinear` active.

## Not proven yet

The following are not proven:

1. BitNet 160M quality.
2. BitNet 160M convergence.
3. Stock-vs-BitNet loss curve comparison.
4. Perplexity or lm-eval results.
5. Checkpoint save/load for BitNet.
6. HuggingFace conversion for BitNet checkpoints.
7. Multi-GPU/FSDP scaling.
8. Performance benefit from quantization.
9. Packed 1.58-bit memory savings.
10. Exact converter accounting.

## Gap: smoke workaround for FSDP

Current state:

- Single-rank FSDP path failed for stock model in the tested environment.
- GPU copy was patched to skip FSDP when all parallelism degrees are 1.

Why this matters:

- TorchTitan is designed for distributed training.
- BitNet scaling claims require distributed correctness.

Next steps:

1. Identify exact TorchTitan commit and expected PyTorch version.
2. Try the recommended environment from TorchTitan docs/CI.
3. Re-run stock debug with unmodified FSDP path.
4. Re-run stock 160M with unmodified FSDP path.
5. Re-run BitNet 160M with unmodified FSDP path.

Success criterion:

- No smoke-only FSDP bypass required.

## Gap: converter count mismatch

Observed:

```text
Swapped 84 Linear layers to BitLinear
bitlinear_count 98
```

Why this matters:

- You need to know exactly which modules were converted.
- Count mismatches can hide accidental over-conversion or under-conversion.

Next instrumentation to add:

1. Print every FQN converted by `BitLinearConverter`.
2. After model build, print every module name whose type is `BitLinear`.
3. Compare the two lists.
4. Add a unit test for expected conversion count on `_160m` config.
5. Explicitly assert `lm_head` stays `Linear`.

Possible command/script idea:

```python
from torchtitan.models.llama3.config_registry import llama3_160m_bitnet
from torchtitan.components.quantization.bitnet import BitLinear

cfg = llama3_160m_bitnet()
model = cfg.model_spec.model.build()
for name, module in model.named_modules():
    if isinstance(module, BitLinear):
        print(name)
```

The actual construction API may differ, so adapt to TorchTitan’s module build path.

## Gap: checkpointing

Smoke disabled checkpointing:

```bash
--checkpoint.interval 0
```

Why it matters:

- Long training needs checkpoints.
- BitLinear parameters and pre-norm weights must save/load correctly.
- HuggingFace conversion may not know what to do with BitLinear.

Next steps:

1. Run 5-10 steps with checkpoint enabled.
2. Resume from the checkpoint.
3. Verify loss continues and no missing/unexpected keys occur.
4. Inspect checkpoint state dict keys for `BitLinear` weights and pre-norm weights.
5. Decide how BitNet checkpoints should convert to HuggingFace.

## Gap: validation

Smoke disabled validation:

```bash
--validator.freq 0
```

Why it matters:

- Training loss alone is not enough.
- Need validation loss/perplexity to compare stock and BitNet.

Next steps:

1. Re-enable validator on a tiny validation schedule.
2. Verify stock 160M validation runs.
3. Verify BitNet 160M validation runs.
4. Record validation loss at identical token budgets.

## Gap: meaningful stock-vs-BitNet curve

A real comparison should control:

- model shape,
- tokenizer/data,
- batch size,
- sequence length,
- number of tokens,
- optimizer,
- learning rate schedule,
- random seed,
- checkpoint/validation frequency,
- hardware/environment.

Minimum next experiment:

- Stock 160M and BitNet 160M.
- Same seed.
- Same token budget.
- Same `seq_len`, `local_batch_size`, and grad accumulation if used.
- Log training loss every step.
- Validate periodically.
- Save checkpoints.

Suggested small next rung:

```text
steps: 100-500
seq_len: 256 or 512
local_batch_size: 1 or 2 depending on memory
validation: every 50 or 100 steps
checkpoint: every 100 or 250 steps
```

Do not jump straight to a huge run until checkpointing and validation are confirmed.

## Gap: performance/memory measurement

This MVP used fake quantization with ordinary `F.linear`. It should not be expected to be faster than stock.

In fact, it may be slower because it adds:

- RMSNorm before each BitLinear,
- activation quantization operations,
- weight quantization operations,
- normal dense matmul anyway.

To measure performance later, track:

- tokens/sec,
- step time,
- GPU memory allocated/reserved,
- model parameter count,
- optimizer state memory,
- activation memory,
- overhead of quantization functions.

## Gap: packed inference / real 1.58-bit memory savings

Current implementation uses full-precision latent weights. It does not pack ternary values.

To get real inference memory savings later, needed pieces include:

1. Ternary packing format.
2. Efficient dequantization or ternary matmul kernel.
3. Export path from trained latent weights to packed ternary weights.
4. Runtime module for packed inference.
5. Accuracy comparison between fake-quant training and packed inference.

This is a later phase.

## Suggested next work order

Recommended order:

1. Add converter/module counting instrumentation.
2. Reconcile `84` vs `98`.
3. Add unit tests for real 160M conversion expectations.
4. Re-enable checkpointing for a short run.
5. Test checkpoint resume.
6. Re-enable validation for a short run.
7. Find/pin a clean TorchTitan/PyTorch environment that does not need the single-rank FSDP workaround.
8. Run a 100-500 step stock-vs-BitNet comparison.
9. Document results in a table.
10. Only then consider longer training or larger model rungs.

## Questions you should be able to answer after reading these docs

If you can answer these, you understand the current MVP:

1. Why does `weight_quant` use an STE?
2. What is the difference between fake quantization and packed 1.58-bit inference?
3. Why is `BitLinear.Config` a subclass of `Linear.Config`?
4. Where does the converter run relative to model construction?
5. Why is `lm_head` skipped?
6. Why did stock debug need to pass before BitNet 160M?
7. Why did the 3-step run finish fast?
8. What did the smoke test prove and not prove?
9. What metadata must be copied when replacing `Linear.Config`?
10. What must be monitored in the next real experiment?

## 2026-06-08 update: next tasks after the completed 100-step shakedown

The suggested 100-step stock-vs-BitNet comparison has now been completed. The next work order should change accordingly:

1. **Validation-focused probe** — force/locate validation metrics and document exactly where TorchTitan emits them.
2. **Checkpoint-resume probe** — resume both stock and BitNet from step-100 or a fresh step-50 checkpoint.
3. **Converter accounting probe** — reconcile `84` logged swaps vs `98` direct `BitLinear` modules.
4. **Clean workaround patch** — make the single-GPU no-FSDP behavior an explicit reproducible option or find an environment where degree-1 FSDP works.
5. **1k-step comparison** — only after validation/resume are understood, run a longer cost-bounded loss curve.

Updated proof state:

- 3-step smoke: done.
- 100-step train/checkpoint shakedown: done.
- validation metric: not yet confirmed.
- resume: not yet confirmed.
- multi-GPU/FSDP: not yet confirmed.
