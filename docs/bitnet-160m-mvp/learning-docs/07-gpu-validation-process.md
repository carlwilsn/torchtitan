# 06 — GPU validation and smoke runs

This page explains what happened on the rented Lambda GPU and why it finished quickly.

## Important: the run was a smoke test

The GPU runs were configured with:

```bash
--training.steps 3
```

That means each run intentionally stopped after only three optimizer steps.

This is why it finished quickly. It was not a full training run. It was a minimum integration proof that the actual training loop can execute.

A real training experiment would use many more steps, larger token budget, checkpointing, validation, and stock-vs-BitNet curves.

## Why the GPU was rented at all

Local Windows validation caught syntax and unit-level issues, but full TorchTitan trainer validation needed Linux GPU because:

- local PyTorch did not match this TorchTitan checkout,
- the trainer imports newer distributed APIs,
- CUDA execution and mixed precision need a real GPU,
- the target use case is scalable training, not only isolated CPU tests.

Local trainer import failed with:

```text
ImportError: cannot import name 'DataParallelMeshDims' from 'torch.distributed.fsdp'
```

That indicated environment mismatch rather than a BitNet unit-test failure.

## Lambda instance and environment

The GPU validation used a Lambda 1x A10 instance.

Recorded environment:

```text
GPU: NVIDIA A10, 23028 MiB
Driver: 570.148.08
Python: 3.10
Torch validation env: ~/venvs/torchtitan-211
PyTorch: 2.11.0+cu128
CUDA: 12.8
```

## Access and lifecycle issues

There were two separate problems:

1. Lifecycle/API tooling for launching and listing Lambda instances.
2. SSH access to the launched instance.

The richer `lambda` tool had a Windows/Git-Bash custom-tool runner path issue. As a workaround, lifecycle control was added to `gpu_run` through a hidden `__lambda__` mode.

Then an instance launched with the wrong SSH key. That meant the API could create the instance, but SSH failed with public key auth errors.

Fix:

- Generate a dedicated local no-passphrase SSH key.
- Register the public key with Lambda.
- Terminate the inaccessible box.
- Relaunch with the matching key.

After that, SSH and `nvidia-smi` worked.

After your cost concern, the lifecycle list was checked and reported no active instances. So the rented GPU is not currently active from the lifecycle tool’s view.

## Code sync to GPU

The local TorchTitan tree was uploaded to the GPU box.

Important detail:

- The local checkout was on Windows.
- Some shell scripts had CRLF line endings.
- On Linux, CRLF can make a shebang look like `/usr/bin/bash^M`.

The GPU copy had to be normalized before `run_train.sh` could execute.

## Environment search

Several PyTorch environment issues appeared.

### Base image PyTorch

The base image had CUDA-working PyTorch, but it lacked newer APIs expected by this TorchTitan checkout, such as:

- `HuggingFaceStorageReader`,
- attention implementation activation helpers,
- `DataParallelMeshDims`.

Patching old PyTorch compatibility one symbol at a time became a bad path.

### Nightly PyTorch

A newer nightly had the needed APIs but hit a backward/storage issue even for stock TorchTitan debug runs.

### Stable PyTorch 2.11

A clean venv with:

```text
torch 2.11.0+cu128
```

had the needed APIs and CUDA support. It was used for validation.

## One-GPU FSDP issue

Even with one GPU and data parallel shard degree set to 1, the stock TorchTitan path still wrapped the model in FSDP. Stock `llama3_debugmodel` failed in backward before any BitNet code was involved:

```text
RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.
```

This is important: the failure happened on stock debug model, so it was not caused by `BitLinear`.

For the smoke ladder only, the GPU copy was patched to skip FSDP wrapping when all parallelism degrees are 1.

This is not a final production fix. It is a one-A10 smoke-test workaround.

## Smoke ladder commands and meaning

### 1. BitNet unit tests on GPU

```bash
source ~/venvs/torchtitan-211/bin/activate
python -m pytest -q tests/unit_tests/test_bitnet_quantization.py
```

Result:

```text
4 passed
```

Meaning:

- The same BitNet unit tests pass in the actual Linux GPU environment.

### 2. Stock debug model

```bash
MODULE=llama3 CONFIG=llama3_debugmodel NGPU=1 LOG_RANK=0 ./run_train.sh \
  --training.steps 3 \
  --activation-checkpoint.mode none \
  --validator.freq 0 \
  --checkpoint.interval 0 \
  --parallelism.data-parallel-shard-degree 1
```

Observed final line:

```text
step: 3  loss: 7.10557  ...  Training completed
```

Meaning:

- The trainer can run a small stock Llama model through forward/backward/optimizer.

### 3. Stock 160M model

```bash
MODULE=llama3 CONFIG=llama3_160m NGPU=1 LOG_RANK=0 ./run_train.sh \
  --training.steps 3 \
  --training.local-batch-size 1 \
  --training.seq-len 128 \
  --activation-checkpoint.mode none \
  --validator.freq 0 \
  --checkpoint.interval 0 \
  --parallelism.data-parallel-shard-degree 1
```

Observed:

```text
Model llama3 160M size: 159,937,536 total parameters
step: 3  loss: 7.41247  ...  Training completed
```

Meaning:

- The new 160M stock config is valid.
- The model shape can run on A10 under short-sequence smoke settings.

### 4. BitNet 160M model

```bash
MODULE=llama3 CONFIG=llama3_160m_bitnet NGPU=1 LOG_RANK=0 ./run_train.sh \
  --training.steps 3 \
  --training.local-batch-size 1 \
  --training.seq-len 128 \
  --activation-checkpoint.mode none \
  --validator.freq 0 \
  --checkpoint.interval 0 \
  --parallelism.data-parallel-shard-degree 1
```

Observed:

```text
Swapped 84 Linear layers to BitLinear
Model llama3 160M size: 160,062,976 total parameters
step: 3  loss: 9.18808  ...  Training completed
```

Meaning:

- The BitNet config converter ran.
- The model built with BitLinear layers.
- Forward/backward/optimizer executed.
- Integration smoke gate passed.

## Why sequence length and batch were reduced

The config default is more ambitious:

```python
local_batch_size=2
seq_len=512
steps=100
```

The smoke run overrode this to:

```bash
--training.local-batch-size 1
--training.seq-len 128
--training.steps 3
```

Reasons:

- A10 has finite memory.
- The purpose was integration, not throughput or convergence.
- Shorter sequence length reduces activation memory.
- Fewer steps reduce cost.

## What was monitored in logs

Key monitored outputs:

- CUDA/GPU visibility through `nvidia-smi`.
- PyTorch version and CUDA version.
- Unit test pass/fail.
- Model parameter count.
- Converter swap count.
- Step/loss line.
- “Training completed”.
- Any traceback before/after BitNet conversion.

## Interpretation of loss values

The reported losses are not meaningful quality measurements yet.

Reasons:

- Only 3 steps.
- Small test dataset/tokenizer path.
- Short sequence length.
- No validation curve.
- Random initialization.

The loss values are useful only as evidence that the loss was finite and training loop progressed.
