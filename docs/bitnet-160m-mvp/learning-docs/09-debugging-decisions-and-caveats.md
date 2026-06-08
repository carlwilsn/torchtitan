# 07 — Debugging log and engineering decisions

This page records why implementation and validation choices were made.

## Decision: use TorchTitan’s converter pattern

Alternative:

- Directly edit Llama block construction to instantiate `BitLinear` instead of `Linear`.

Chosen:

- Add `BitLinearConverter` that rewrites `Linear.Config` to `BitLinear.Config` before model build.

Why:

1. It mirrors existing Float8/MXFP8 integration style.
2. It keeps stock model code unchanged.
3. It allows stock-vs-BitNet configs to share the same model flavor.
4. It gives an easy filtering mechanism for `lm_head` or other modules.
5. It scales better to future experiments.

Learning point:

- In config-driven systems, model surgery often happens at the config tree level before modules exist.

## Decision: fake quantization with STE, not packed ternary kernels

Alternative:

- Implement packed ternary storage and custom kernels immediately.

Chosen:

- Keep latent full-precision weights and use fake quantization in forward.

Why:

1. Autograd works through standard PyTorch operations.
2. Optimizers can update normal parameters.
3. TorchTitan module integration is easier.
4. The first goal is training-stack integration, not deployment efficiency.
5. Packed kernels would introduce a second large project before proving the model path.

Learning point:

- Training quantization and inference quantization are related but not the same engineering problem.

## Decision: leave `lm_head` unquantized

Alternative:

- Quantize every linear layer, including output projection.

Chosen:

- Default filter skips `output` and `lm_head`.

Why:

1. Output logits are sensitive.
2. Keeping `lm_head` stock reduces early instability.
3. It matches common quantization practice to preserve some boundary layers.
4. It makes debugging easier because final vocabulary projection remains ordinary `Linear`.

Open question:

- A later ablation should test quantizing the output head.

## Decision: add a local 160M flavor using small vocab

Alternative:

- Use existing 1B/3B flavors or real Llama vocab/assets immediately.

Chosen:

- Add `160M` with `vocab_size=2048` using `tests/assets/tokenizer`.

Why:

1. Avoid gated Llama assets.
2. Fit on one A10 for smoke testing.
3. Make experiments cheap.
4. Exercise nontrivial Llama blocks without jumping to full scale.

Tradeoff:

- This is not exactly a real Llama tokenizer/vocab setup. It is a training-stack rung.

## Decision: copy `param_init` and `sharding_config` in converter

Alternative:

- Create `BitLinear.Config` with only dimensions and BitNet flags.

Chosen:

- Copy structural and training metadata from original `Linear.Config`.

Why:

1. Initialization should match stock model as much as possible.
2. Distributed/sharding metadata should not be lost.
3. The converted model should be equivalent except for the linear implementation.

Learning point:

- A converter must preserve metadata, not just dimensions.

## Decision: add `BitLinear.Config` to `has_quantization`

Alternative:

- Only define the module and converter.

Chosen:

- Update quantization detection utilities.

Why:

1. TorchTitan uses config inspection for quantization-aware behavior.
2. If BitNet is invisible to that utility, later features may silently skip quantization paths.
3. This keeps BitNet aligned with existing Float8/MXFP8 design.

## Debugging: local Windows environment mismatch

Symptom:

```text
ImportError: cannot import name 'DataParallelMeshDims' from 'torch.distributed.fsdp'
```

Interpretation:

- The installed local PyTorch did not match this TorchTitan checkout’s expected APIs.

Decision:

- Do not treat this as a BitNet implementation failure.
- Move full trainer validation to Linux GPU with a compatible PyTorch environment.

Learning point:

- Unit tests can pass while full framework imports fail due to environment version mismatch.

## Debugging: Lambda lifecycle/tooling issue

Symptom:

- `gpu_run` connectivity existed, but no active instance.
- Richer `lambda` lifecycle tool failed before executing its script due to Windows/POSIX path translation in the custom-tool runner.

Decision:

- Add Lambda lifecycle workaround through `gpu_run` hidden control mode.

Why:

- Need to list regions, SSH keys, launch, and terminate instances without waiting for tool runtime reload.

Learning point:

- Infrastructure tooling can block ML progress even when model code is ready.

## Debugging: SSH key mismatch

Symptom:

```text
Permission denied (publickey)
```

Cause:

- Instance was launched with a Lambda SSH key that did not match available local private keys.

Fix:

1. Generate a dedicated local no-passphrase key.
2. Register its public key with Lambda.
3. Terminate inaccessible instance.
4. Relaunch with the matching key.

Learning point:

- Cloud instance launch success does not imply SSH access success. Key registration must match the local private key.

## Debugging: CRLF line endings

Symptom:

- Linux tried to run a shell interpreter path ending in `^M`, such as `/usr/bin/bash^M`.

Cause:

- Windows CRLF line endings in uploaded scripts.

Fix:

- Normalize line endings on the GPU copy before running `run_train.sh`.

Learning point:

- Windows-to-Linux sync can break shell scripts even when Python files are fine.

## Debugging: PyTorch version search

### Base image

Pros:

- CUDA worked.

Cons:

- Missing newer TorchTitan-required APIs.

### Nightly

Pros:

- Had newer APIs.

Cons:

- Hit backward/storage failures even for stock model.

### Clean PyTorch 2.11 venv

Pros:

- CUDA worked.
- Required APIs were mostly present.
- Supported successful smoke after single-rank FSDP workaround.

Chosen:

- Use clean `~/venvs/torchtitan-211` for validation.

Learning point:

- For fast-moving training frameworks, exact PyTorch version matters as much as model code.

## Debugging: stock one-GPU FSDP failure

Symptom:

```text
RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.
```

Important observation:

- This happened in stock `llama3_debugmodel` before any BitNet code was involved.

Interpretation:

- The issue was in TorchTitan/PyTorch/FSDP interaction, not BitNet.

Smoke workaround:

- On the GPU copy only, skip FSDP wrapping when all parallelism degrees are 1.

Why acceptable for smoke:

- The goal was single-GPU integration proof.
- With one GPU and no sharding, FSDP is not needed to prove forward/backward/optimizer.

Why not final:

- Real scaling needs FSDP/multi-GPU validation.
- The workaround should not be confused with a production distributed training fix.

## Decision: disable validation/checkpointing during smoke

Smoke commands included:

```bash
--validator.freq 0
--checkpoint.interval 0
```

Why:

1. Reduce moving parts.
2. Avoid testing checkpoint conversion before basic training works.
3. Reduce cost and runtime.
4. Make failures easier to interpret.

Later:

- Re-enable these after stable longer training path.

## Decision: use short sequence length and batch size

Smoke overrides:

```bash
--training.local-batch-size 1
--training.seq-len 128
```

Why:

1. Avoid A10 memory pressure.
2. Make test fast.
3. Prove integration cheaply.

Tradeoff:

- Does not measure realistic throughput or quality.

## Decision: document count mismatch instead of hiding it

Observed:

- Converter log: `Swapped 84 Linear layers to BitLinear`.
- Direct model-construction count: `bitlinear_count 98`.

Decision:

- Record the mismatch as a known gap.

Why:

- Smoke success is real, but accounting must be reconciled before claiming converter behavior is fully understood.
- This is exactly the kind of detail that matters for learning ownership.

Possible explanations to investigate:

1. Counting configs vs instantiated modules differs.
2. Some `BitLinear` modules are created from reused/shared configs.
3. Direct count may include internal/pre-norm-related modules or duplicated traversal behavior.
4. Training log and direct check may have used slightly different construction paths.
5. Weight tying or output head handling affects counts.

The correct response is not to guess; add explicit tests/instrumentation.
