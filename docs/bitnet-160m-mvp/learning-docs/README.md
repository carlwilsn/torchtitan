# BitNet 160M MVP learning documentation

This folder is the detailed learning record for the TorchTitan BitNet b1.58 MVP. It is intentionally more verbose than the project README.

The goal is to let you reconstruct what was built, why it was built that way, what files were touched, what each function/data structure does, what was monitored, what failed, and what the GPU smoke result actually proves.

## Recommended reading order

1. [Goal and mental model](01-goal-and-mental-model.md)
2. [TorchTitan architecture points used](02-torchtitan-architecture.md)
3. [BitNet math and implementation](03-bitnet-math-and-bitlinear.md)
4. [Converter and model surgery](04-converter-and-config-tree.md)
5. [160M model and training configs](05-160m-model-and-training-configs.md)
6. [Tests and local validation](06-tests-and-local-validation.md)
7. [GPU validation process](07-gpu-validation-process.md)
8. [Training loop and monitored structures](08-training-loop-and-monitored-structures.md)
9. [Debugging decisions and caveats](09-debugging-decisions-and-caveats.md)
10. [Next experiments and learning checklist](10-next-experiments.md)

## One-sentence status

The MVP has a working TorchTitan path where a Llama-style 160M config can be converted to use BitNet-style `BitLinear` layers and can complete a short forward/backward/optimizer smoke run on a Lambda A10 GPU; it has not yet run a long training experiment or proven paper-scale quality.

## Important cost note

The Lambda A10 used for validation was terminated after the smoke tests. The smoke tests were short because they used only 3 training steps, small sequence length, batch size 1, and disabled validation/checkpointing. That is why the run completed quickly.
