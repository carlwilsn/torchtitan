# BitNet 160M TorchTitan MVP

This folder documents the first TorchTitan-based BitNet reproduction rung.

The goal is **not** to observe the paper's headline 3B-scale result yet. The goal is to make a small, end-to-end, scalable training path that teaches and exercises the same TorchTitan mechanisms needed later: model config registry, converter-based model surgery, FSDP-compatible module configs, dataloading, metrics, checkpointing, and launch/debug flow.

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

## GPU gate

At the time this doc was written:

- `gpu_run` works as a connectivity tool, but reported no active Lambda instances.
- The richer `lambda` lifecycle tool was present but the loaded custom-tool runner failed before executing the shim path.
- The local `run.sh` shim was patched for Git-Bash/Windows path compatibility, but the already-loaded tool still failed until the custom tool runtime is reloaded or otherwise fixed.

So the code path is ready for GPU integration testing, but the actual stock/BitNet TorchTitan training runs are still pending GPU access.
