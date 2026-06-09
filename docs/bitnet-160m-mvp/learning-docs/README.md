# BitNet 160M MVP learning documentation

This folder is the detailed learning record for the TorchTitan BitNet b1.58 MVP. It is intentionally more verbose than the project README.

The goal is to let you reconstruct what was built, why it was built that way, what files were touched, what each function/data structure does, what was monitored, what failed, and what the GPU experiments actually prove.

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
11. [Reading the 100-step shakedown like a scientist](11-100-step-shakedown-analysis.md)
12. [The 1700-step seed-locked stock-vs-BitNet run](12-1700-step-seedlocked-comparison.md)

## One-sentence status

The MVP has a working TorchTitan path where a Llama-style 160M config can be converted to use BitNet-style `BitLinear` layers. It passed both a 3-step GPU smoke test and a 100-step stock-vs-BitNet shakedown on a Lambda A10. It has not yet proven paper-scale quality, multi-GPU behavior, validation-loss correctness, checkpoint resume, or packed 1.58-bit inference benefits.

## Experiment progression

| Stage | Purpose | Result |
| --- | --- | --- |
| Local unit tests | Prove quantization functions, STE gradients, `BitLinear`, and converter behavior in isolation | Passed: `4 passed` |
| Local syntax checks | Catch import/syntax errors in changed files | Passed |
| 3-step GPU smoke | Prove stock and BitNet configs enter the real TorchTitan train loop | Passed |
| 100-step shakedown | Prove a bounded stock-vs-BitNet experiment loop with checkpoints and artifact collection | Passed |
| 1700-step seed-locked run | First real comparison: seed-locked, real c4, ~55.7M tokens/run, validation emitting loss/ppl | Passed; +0.13-nat FP-vs-ternary gap (1 seed) |

Completed experiment records:

- [`../experiments/2026-06-08-100-step-shakedown/README.md`](../experiments/2026-06-08-100-step-shakedown/README.md)
- [`../results/README.md`](../results/README.md) — 2026-06-09 1700-step seed-locked run

## Important cost note

The Lambda A10 instances used for validation were terminated after the runs. The first smoke tests were short because they used only 3 training steps, small sequence length, batch size 1, and disabled validation/checkpointing. The later 100-step shakedown also used small sequence length and batch size, and the GPU was terminated after logs/artifacts were collected.
