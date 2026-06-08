# 03 — TorchTitan integration map

This page explains where each file fits into TorchTitan’s training stack.

## High-level data/control flow

The BitNet 160M smoke run begins with a shell command like:

```bash
MODULE=llama3 CONFIG=llama3_160m_bitnet NGPU=1 LOG_RANK=0 ./run_train.sh ...
```

The rough control flow is:

1. `run_train.sh` launches the TorchTitan training entrypoint.
2. `MODULE=llama3` selects the Llama3 model package.
3. `CONFIG=llama3_160m_bitnet` selects a function in `config_registry.py`.
4. The config function returns a `Trainer.Config` object.
5. `Trainer.Config.model_spec` is built by `model_registry("160M", converters=[...])`.
6. `model_registry` builds a `Llama3Model.Config` tree.
7. `BitLinearConverter` traverses that config tree and replaces selected `Linear.Config` nodes with `BitLinear.Config` nodes.
8. The trainer builds the actual model modules from the modified config tree.
9. During model construction, `BitLinear.Config` nodes instantiate `BitLinear` modules.
10. During each forward pass, `BitLinear.forward()` applies optional RMSNorm, activation quantization, weight quantization, and `F.linear`.
11. Loss, backward, optimizer, metrics, and scheduler proceed through normal TorchTitan components.

## Files changed

### `torchtitan/components/quantization/bitnet.py`

This is the core BitNet implementation.

Defines:

- `activation_quant`
- `weight_quant`
- `BitLinear`
- `BitLinearConverter`

Why here:

- TorchTitan already has quantization components under `components/quantization`.
- Existing Float8/MXFP8 code uses converter-style integration.
- Keeping BitNet here makes it a peer of other quantization paths.

### `torchtitan/components/quantization/__init__.py`

This exports the BitNet symbols from the quantization package.

Why this matters:

`config_registry.py` imports:

```python
from torchtitan.components.quantization import BitLinearConverter, Float8LinearConverter
```

That only works if `BitLinearConverter` is re-exported in `__init__.py`.

### `torchtitan/components/quantization/utils.py`

This file contains quantization helper logic.

The relevant change is in `has_quantization(model_config)`. It now treats `BitLinear.Config` as a quantized linear config.

Why this matters:

TorchTitan uses config-tree inspection to decide whether quantization-specific paths are active. If BitNet configs are not recognized there, downstream logic may behave as if the model is unquantized.

### `torchtitan/models/llama3/__init__.py`

This file defines Llama model “flavors”: debug, 1B, 3B, 8B, etc.

Added:

```python
"160M": _160m
```

and the `_160m` function.

Why this matters:

TorchTitan separates model shape from training recipe. A model flavor describes the architecture: dimensions, heads, layers, vocab, embeddings, rope, and transformer blocks.

### `torchtitan/models/llama3/config_registry.py`

This file defines named training configs selected by `CONFIG=...`.

Added:

- `llama3_160m()`
- `llama3_160m_bitnet()`

Why this matters:

The config registry is the user-facing entrypoint for running experiments. The stock and BitNet configs must be separate so we can compare them.

### `tests/unit_tests/test_bitnet_quantization.py`

This file tests the new BitNet implementation without needing the full trainer.

Why this matters:

Small unit tests isolate bugs in quantization and conversion before expensive GPU runs.

## Key TorchTitan objects involved

### `Trainer.Config`

`Trainer.Config` is the top-level recipe for a training run. It includes:

- model spec,
- loss,
- optimizer,
- scheduler,
- training steps and batch shape,
- dataloader,
- metrics,
- checkpointing,
- activation checkpointing,
- validation,
- parallelism.

The 160M configs return this object.

### `ModelSpec`

`ModelSpec` packages model information for TorchTitan:

- name,
- flavor,
- model config tree,
- parallelization function,
- pipelining function,
- state dict adapter.

The BitNet converter acts before `ModelSpec` is returned.

### `Llama3Model.Config`

This is a nested dataclass-like config tree for the Llama model.

It contains:

- embedding config,
- final norm config,
- output head config,
- rope config,
- list of transformer block configs.

The converter traverses this tree looking for `Linear.Config` nodes.

### `Linear.Config`

TorchTitan’s common `Linear` module is config-driven. A linear config stores:

- `in_features`,
- `out_features`,
- `bias`,
- parameter initialization,
- sharding config.

The converter replaces selected instances of this with `BitLinear.Config`.

### `BitLinear.Config`

`BitLinear.Config` subclasses `Linear.Config`, adding BitNet-specific knobs:

- `activation_quant`,
- `weight_quant`,
- `pre_norm`,
- `eps`.

Subclassing `Linear.Config` is a key design choice because it lets the config-tree traversal find BitNet configs through the same `Linear.Config` type hierarchy.

## Why use a converter instead of editing every layer manually?

Manual editing would mean changing Llama block construction code to instantiate BitLinear everywhere. That is brittle and makes experiments hard to toggle.

The converter approach is better because:

1. It keeps the base Llama architecture unchanged.
2. It lets stock and BitNet configs share one model flavor.
3. It matches TorchTitan’s existing quantization design.
4. It can filter which modules are converted.
5. It makes future experiments easy: change converter settings, not the model code.

## Why keep `lm_head` unquantized?

The converter default filter skips fully qualified names containing:

```python
["output", "lm_head"]
```

The output projection is often left higher precision in quantized LLM experiments because it is directly responsible for logits over the vocabulary. Quantizing it can hurt stability and makes early debugging harder.

For this MVP, leaving `lm_head` alone reduces risk. It also gives a clear thing to test later: compare with and without output-head quantization.
