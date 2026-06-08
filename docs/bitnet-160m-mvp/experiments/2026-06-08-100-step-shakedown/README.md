# 2026-06-08 — 100-step stock-vs-BitNet shakedown

## Goal

Run the first cost-bounded experiment after the 3-step smoke test.

This is still **not** full pretraining. It is a full experiment loop:

1. launch a rented GPU,
2. prepare a reproducible environment,
3. run stock 160M and BitNet 160M with matched settings,
4. test whether checkpointing can be enabled,
5. collect logs/artifacts,
6. terminate the GPU,
7. write down what happened and why.

The learning goal is to move from “does it run for 3 steps?” to “can we operate this as an experiment with logs, checkpoints, and a stock-vs-BitNet comparison?”

## Cost guardrail

Use one Lambda `gpu_1x_a10` instance and terminate it when the loop is done or when a blocking failure appears. Lambda listed this instance type at about `$1.29/hour` at launch time.

The instance was terminated after the artifacts were collected. A final Lambda lifecycle check showed no active instances.

## Environment

Remote validation used a Lambda 1x A10 instance:

```text
GPU: NVIDIA A10, 23028 MiB
Driver: 570.148.08
Python venv: ~/venvs/torchtitan-211
PyTorch: 2.11.0+cu128
CUDA available: true
```

Two remote-only compatibility/workaround patches were used, matching the earlier smoke validation:

1. **Torch 2.11 import compatibility:** provide narrow fallbacks for newer PyTorch symbols imported by this TorchTitan checkout.
2. **Single-GPU FSDP bypass:** skip FSDP wrapping when all parallelism degrees are effectively one. This avoids the degree-1 FSDP/meta-tensor backward failure seen in stock TorchTitan before any BitNet code is involved.

These patches were applied on the rented GPU copy for the experiment. They should be made explicit and reproducible before longer experiments.

## Commands

Both runs used the same model shape and training knobs except for the BitNet converter.

Common overrides:

```bash
--training.steps 100
--training.local-batch-size 1
--training.seq-len 128
--activation-checkpoint.mode none
--parallelism.data-parallel-shard-degree 1
--checkpoint.enable
--checkpoint.interval 50
--checkpoint.keep-latest-k 2
--validator.freq 50
--validator.steps 2
```

Stock:

```bash
MODULE=llama3 CONFIG=llama3_160m NGPU=1 LOG_RANK=0 ./run_train.sh \
  --training.steps 100 \
  --training.local-batch-size 1 \
  --training.seq-len 128 \
  --activation-checkpoint.mode none \
  --parallelism.data-parallel-shard-degree 1 \
  --checkpoint.enable \
  --checkpoint.interval 50 \
  --checkpoint.keep-latest-k 2 \
  --validator.freq 50 \
  --validator.steps 2
```

BitNet:

```bash
MODULE=llama3 CONFIG=llama3_160m_bitnet NGPU=1 LOG_RANK=0 ./run_train.sh \
  --training.steps 100 \
  --training.local-batch-size 1 \
  --training.seq-len 128 \
  --activation-checkpoint.mode none \
  --parallelism.data-parallel-shard-degree 1 \
  --checkpoint.enable \
  --checkpoint.interval 50 \
  --checkpoint.keep-latest-k 2 \
  --validator.freq 50 \
  --validator.steps 2
```

## Results summary

Status: **completed**.

Both matched 100-step runs completed without crashing. Checkpointing saved checkpoints at steps 50 and 100 for both stock and BitNet.

| Run | Steps logged | Completed? | Step 1 loss | Step 100 loss | First 10 avg loss | Last 10 avg loss | Avg tokens/sec | Last 10 avg tokens/sec | Max logged GPU memory | Checkpoints |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stock `llama3_160m` | 100 | yes | 8.12197 | 2.78636 | 7.61875 | 2.86142 | 3571.7 | 3652.3 | 1.28 GiB | step-50, step-100 |
| BitNet `llama3_160m_bitnet` | 100 | yes | 7.99145 | 2.75976 | 5.67436 | 2.82334 | 1875.9 | 1922.5 | 1.37 GiB | step-50, step-100 |

Important interpretation notes:

- The loss decreased in both runs over 100 steps.
- The two losses are **not evidence that BitNet is better**. This is a tiny run on tiny sequence length, and randomness/data order dominate at this scale.
- BitNet was slower in this naive implementation: about `1876` average tokens/sec vs `3572` for stock. That is expected because this MVP uses simple fake quantization in regular PyTorch ops, not packed ternary kernels.
- BitNet used slightly more logged memory here (`1.37 GiB` vs `1.28 GiB`), also expected for this training-time fake-quant implementation because full-precision weights remain present and quantized temporary tensors are created.

## Raw log evidence

Stock model setup:

```text
Model llama3 160M size: 159,937,536 total parameters
```

Stock final status:

```text
step: 100  loss: 2.78636  grad_norm: 3.8125  memory: 1.28GiB  tps: 3,565
Training completed
Elapsed (wall clock) time: 0:16.93
Exit status: 0
```

BitNet model setup:

```text
Swapped 84 Linear layers to BitLinear
Model llama3 160M size: 160,062,976 total parameters
```

BitNet final status:

```text
step: 100  loss: 2.75976  grad_norm: 2.5156  memory: 1.37GiB  tps: 1,912
Training completed
Elapsed (wall clock) time: 0:20.18
Exit status: 0
```

Checkpoint evidence:

```text
stock_checkpoints/step-50/.metadata
stock_checkpoints/step-50/__0_0.distcp
stock_checkpoints/step-100/.metadata
stock_checkpoints/step-100/__0_0.distcp

bitnet_checkpoints/step-50/.metadata
bitnet_checkpoints/step-50/__0_0.distcp
bitnet_checkpoints/step-100/.metadata
bitnet_checkpoints/step-100/__0_0.distcp
```

## Validation status

The runs included:

```bash
--validator.freq 50
--validator.steps 2
```

However, the captured logs do **not** show clear validation metric lines. Therefore the honest conclusion is:

- checkpointing worked,
- the train loop worked,
- stock and BitNet both completed 100 optimizer steps,
- validation was configured but no validation metric was observed in the captured logs.

Before using validation loss as a decision metric, run a dedicated validation-focused probe and confirm exactly where TorchTitan emits validation results for this config/dataset.

## Artifacts

Local artifact folder:

```text
docs/bitnet-160m-mvp/experiments/2026-06-08-100-step-shakedown/artifacts/
```

Kept artifacts:

- `environment.txt` — remote environment summary.
- `stock_100.log` — raw stock run log.
- `bitnet_100.log` — raw BitNet run log.
- `stock_checkpoint_files.txt` — checkpoint file listing, not full checkpoint payload.
- `bitnet_checkpoint_files.txt` — checkpoint file listing, not full checkpoint payload.
- `stock_structured_logs/` — TorchTitan structured JSONL logs.
- `bitnet_structured_logs/` — TorchTitan structured JSONL logs.
- `stock_loss_metrics.csv` — parsed per-step stock metrics.
- `bitnet_loss_metrics.csv` — parsed per-step BitNet metrics.
- `parsed_summary.json` — parsed summary numbers used in this document.

Full checkpoint payloads were **not** kept locally because they are large and unnecessary for documentation. Only file listings were retained.

## What this experiment proves

This experiment proves more than the 3-step smoke test:

1. The stock 160M TorchTitan path can complete 100 one-GPU training steps with the single-rank workaround.
2. The BitNet 160M TorchTitan path can complete the same 100-step loop.
3. BitNet fake-quant training does not immediately diverge or crash over this short run.
4. Checkpointing can save step-50 and step-100 checkpoints for both variants.
5. The experiment loop can provision GPU, sync code, run, collect artifacts, and terminate.

## What this experiment does not prove

This is still not paper-scale or quality evidence.

It does **not** prove:

- BitNet has matched full-precision quality.
- BitNet is faster in this implementation.
- BitNet uses less memory in this implementation.
- Validation loss is working and meaningful.
- Multi-GPU/FSDP behavior works.
- Checkpoint resume works.
- Longer runs remain stable.
- The `84` logged swap count vs `98` direct module count mismatch is resolved.

## Follow-up questions answered

- Did checkpoint save work for stock and BitNet? **Yes.** Step 50 and step 100 checkpoint files were created for both.
- Is BitNet loss decreasing over 100 steps? **Yes, in this tiny run.** Step 1 loss was `7.99145`; step 100 loss was `2.75976`; last-10-step average was `2.82334`.
- How different are stock and BitNet step speeds? **BitNet was slower.** Average tokens/sec was about `1876` for BitNet vs `3572` for stock.
- Does memory stay stable? **Yes at this scale.** Logged memory stayed around `1.28 GiB` stock and `1.37 GiB` BitNet.
- Did validation work? **Unclear.** It was configured but no validation metric was observed in logs.
- Does the converter count mismatch matter operationally? **Not for this 100-step run, but yes for understanding and final correctness claims.**

## Next step

Before spending on a longer run, do one of these:

1. Run a validation-focused probe and confirm validation metrics are emitted.
2. Run a checkpoint-resume probe from the step-100 BitNet checkpoint.
3. Reconcile the `84` vs `98` BitLinear count mismatch.
4. Make the single-GPU workaround a clean, documented local patch or find the TorchTitan/PyTorch combination where degree-1 FSDP works without workaround.
5. Then run a 1k-step stock-vs-BitNet comparison with the same artifact discipline.
