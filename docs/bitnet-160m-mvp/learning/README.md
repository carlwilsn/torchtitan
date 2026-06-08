# BitNet 160M MVP learning documentation

This folder is the deep learning-oriented explanation of the BitNet 160M TorchTitan MVP. It is written for understanding, not just for remembering which commands passed.

The short README one directory up records the status. These notes explain what was built, why each piece exists, how data flows through it, what was tested, what failed, and what remains uncertain.

## Reading order

1. [Goal, scope, and what “done” means](01-goal-scope-and-status.md)
2. [BitNet b1.58 mechanics implemented here](02-bitnet-layer-mechanics.md)
3. [TorchTitan integration points and files](03-torchtitan-integration-map.md)
4. [Function-by-function implementation walkthrough](04-function-by-function.md)
5. [Testing strategy and monitored structures](05-tests-and-invariants.md)
6. [GPU validation, environment, and smoke runs](06-gpu-validation-and-smoke-runs.md)
7. [Debugging log and engineering decisions](07-debugging-decisions.md)
8. [Known gaps and next experiments](08-known-gaps-and-next-experiments.md)

## One-sentence summary

We added a BitNet-style `BitLinear` module and a TorchTitan config converter that swaps Llama internal `Linear.Config` entries to `BitLinear.Config`, then proved on a Lambda A10 that stock and BitNet 160M configurations can execute forward, backward, and optimizer steps in TorchTitan.

## Important correction: smoke test is not training completion

The GPU runs were intentionally tiny **3-step smoke tests**. They answer this question:

> Does the integrated TorchTitan model build and execute one or more real training iterations without crashing?

They do **not** answer:

> Has the 160M model been trained to convergence or shown BitNet quality/performance benefits?

A 3-step smoke test can finish quickly because it uses only a few batches, short sequence length, and no checkpoint/validation work. A real comparison needs thousands to millions of tokens, longer curves, checkpointing, validation, and careful stock-vs-BitNet measurement.
