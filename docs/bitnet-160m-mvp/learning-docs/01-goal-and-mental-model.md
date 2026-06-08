# 01 — Goal, scope, and status

## Original learning goal

The larger project goal is to reproduce and understand BitNet b1.58-style training, eventually at meaningful scale, while preserving your own understanding of the implementation. The immediate TorchTitan goal was smaller and more concrete:

1. Add a BitNet-style quantized linear layer to TorchTitan.
2. Wire it into TorchTitan using the same config/converter pattern used by existing quantization paths.
3. Create a 160M-scale Llama-style rung that is small enough to run on one rented A10.
4. Prove the stock and BitNet configurations can execute the actual TorchTitan training loop on Linux GPU.
5. Document what was built so you can reconstruct it and modify it yourself.

## What was actually completed

Implemented locally in this checkout:

- BitNet quantization functions for activations and weights.
- A `BitLinear` module that replaces TorchTitan’s common `Linear` module.
- A `BitLinearConverter` that rewrites the model config tree before the model is instantiated.
- A new `160M` Llama-style model flavor.
- Two config registry entries:
  - `llama3_160m` for stock baseline.
  - `llama3_160m_bitnet` for BitNet conversion.
- Unit tests covering quantization behavior, gradient flow, module forward/backward, and converter replacement.
- GPU smoke validation on Lambda A10.

## What “passed” means

The GPU result was an integration smoke pass, not a full training result.

The smoke ladder proved:

1. The TorchTitan trainer can import and launch in a compatible GPU environment.
2. The stock debug model can run training steps.
3. The stock 160M model can run training steps.
4. The BitNet 160M model can run training steps with `BitLinear` active.

A training step includes:

1. Loading or generating a batch from the dataloader.
2. Forward pass through the model.
3. Computing language-model loss.
4. Backward pass through the graph.
5. Optimizer update.
6. Metrics logging.

## What did not happen yet

The following are still open:

- No long BitNet training curve was run.
- No validation perplexity comparison was produced.
- No checkpoint was converted to HuggingFace format.
- No lm-eval benchmark was run.
- No packed 1.58-bit inference format was implemented.
- No multi-GPU/FSDP scaling validation was completed.
- The converter count mismatch (`84` logged vs `98` direct module count) has not been reconciled.

## Why this stage matters anyway

Before this MVP, the project had BitNet-like code in isolation. After this MVP, BitNet is inside TorchTitan’s actual model/config/training machinery. That matters because the difficult scaling work later depends on the same surfaces:

- model config registry,
- module config trees,
- converter-based model surgery,
- trainer construction,
- dataloader and loss integration,
- distributed wrapping,
- optimizer and scheduler setup,
- metrics and checkpoint hooks.

The MVP is the first end-to-end rung on the path from “custom layer works” to “train a BitNet LLM in a real training stack.”

## Current cost state

The rented Lambda GPU was used for short smoke tests. After your cost concern, the lifecycle tool was checked and reported no active instances. The GPU is not currently running from the lifecycle tool’s view.
