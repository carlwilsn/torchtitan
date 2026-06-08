# BitNet 160M TorchTitan MVP

This folder documents the first TorchTitan-based BitNet reproduction rung.

The goal is **not** to observe the paper's headline 3B-scale result yet. The goal is to make a small, end-to-end, scalable training path that teaches and exercises the same TorchTitan mechanisms needed later: model config registry, converter-based model surgery, FSDP-compatible module configs, dataloading, metrics, checkpointing, and launch/debug flow.

For a detailed learning walkthrough of what was built, why it was built that way, how each file/function works, what was monitored, what failed, and what remains next, read:

- [learning-docs/README.md](learning-docs/README.md) — index for the multi-file implementation walkthrough.
- [experiments/README.md](experiments/README.md) — controlled follow-up experiment records after the initial smoke test.

## Current implementation state

Implemented in this checkout:

- `torchtitan/components/quantization/bitnet.py`
  - `activation_quant`: per-token absmax int8 activation quantization with identity STE.
  - `weight_quant`: absmean ternary weight quantization with identity STE.
  - `BitLinear`: TorchTitan `Module`-protocol-compatible Linear replacement.
  - `BitLinearConverter`: config-tree converter that swaps `Linear.Config` to `BitLinear.Config`.
- `torchtitan/components/quantization/__init__.py`
  - Exports the BitNet classes/functions next to Float8/MXFP8.
- `torchtitan/components/quantization/utils.py`
  - Makes `has_quantization()` recognize `BitLinear.Config`.
- `torchtitan/models/llama3/__init__.py`
  - Adds a `160M` Llama-style flavor using the local test tokenizer vocabulary.
- `torchtitan/models/llama3/config_registry.py`
  - Adds `llama3_160m` stock config.
  - Adds `llama3_160m_bitnet` config using `BitLinearConverter`.
- `tests/unit_tests/test_bitnet_quantization.py`
  - Unit tests for quantization values, STE gradient flow, BitLinear forward/backward, and converter replacement.

## Validation completed locally

Local Windows validation completed:

```bash
python -m pytest -q tests/unit_tests/test_bitnet_quantization.py
# 4 passed

python -m py_compile \
  torchtitan/components/quantization/bitnet.py \
  torchtitan/components/quantization/__init__.py \
  torchtitan/components/quantization/utils.py \
  torchtitan/models/llama3/__init__.py \
  torchtitan/models/llama3/config_registry.py \
  tests/unit_tests/test_bitnet_quantization.py
# passed
```

Full TorchTitan trainer import/build tests could not be completed on local Windows because the installed local PyTorch does not match this TorchTitan checkout. The failure was:

```text
ImportError: cannot import name 'DataParallelMeshDims' from 'torch.distributed.fsdp'
```

That is an environment-version issue, not a BitLinear unit-test failure. Full integration must be validated on a Linux GPU box with the TorchTitan-compatible PyTorch build.

## GPU validation completed

Validated on 2026-06-08 on a Lambda 1x A10 instance:

```text
GPU: NVIDIA A10, 23028 MiB
Driver: 570.148.08
Python: 3.10
Torch validation env: ~/venvs/torchtitan-211
PyTorch: 2.11.0+cu128
CUDA: 12.8
```

The uploaded Windows checkout had CRLF line endings, so the GPU copy was normalized before launch.

The stock TorchTitan one-GPU path initially failed before any BitNet code was involved:

```text
RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet.
```

This happened in backward even for stock `llama3_debugmodel`, with activation checkpointing disabled. For the one-A10 smoke ladder, the GPU copy was patched to skip FSDP wrapping when all parallelism degrees are 1. This is a smoke-test workaround, not a replacement for later multi-GPU/FSDP validation.

### GPU smoke commands

BitNet unit tests passed on the GPU environment:

```bash
source ~/venvs/torchtitan-211/bin/activate
python -m pytest -q tests/unit_tests/test_bitnet_quantization.py
# 4 passed
```

Stock debug model passed 3 training steps:

```bash
MODULE=llama3 CONFIG=llama3_debugmodel NGPU=1 LOG_RANK=0 ./run_train.sh \
  --training.steps 3 \
  --activation-checkpoint.mode none \
  --validator.freq 0 \
  --checkpoint.interval 0 \
  --parallelism.data-parallel-shard-degree 1
```

Observed final log line:

```text
step: 3  loss: 7.10557  ...  Training completed
```

Stock 160M passed 3 training steps:

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

Observed model size and final log line:

```text
Model llama3 160M size: 159,937,536 total parameters
step: 3  loss: 7.41247  ...  Training completed
```

BitNet 160M passed 3 training steps:

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

Observed model size and final log line:

```text
Swapped 84 Linear layers to BitLinear
Model llama3 160M size: 160,062,976 total parameters
step: 3  loss: 9.18808  ...  Training completed
```

A direct model-construction check confirmed the output projection is left unquantized:

```text
bitlinear_count 98
lm_head_type Linear
tok_embeddings_type Embedding
```

The training-time converter log (`84`) and direct post-build module count (`98`) do not yet agree. This does not block the smoke result, but it should be reconciled before claiming the converter accounting is final.

## Current status

The MVP now has an end-to-end GPU smoke path:

1. TorchTitan imports and launches on A10 with a compatible PyTorch environment.
2. Stock debug training runs forward/backward/optimizer.
3. Stock 160M training runs forward/backward/optimizer.
4. BitNet 160M training runs forward/backward/optimizer with `BitLinear` layers active.

Remaining work before a real scaling experiment:

- Remove or upstream the single-rank FSDP smoke workaround by validating on a TorchTitan/PyTorch combination where stock degree-1 FSDP works.
- Reconcile the BitLinear converter swap count (`84` logged vs `98` direct modules).
- Re-enable activation checkpointing, validation, and checkpointing after the smoke path is stable.
- Run longer stock-vs-BitNet loss curves at a meaningful sequence length/batch.
