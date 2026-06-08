# 2026-06-08 — 100-step stock-vs-BitNet shakedown

## Goal

Run the first cost-bounded experiment after the 3-step smoke test.

This is still **not** full pretraining. It is a full experiment loop:

1. launch a rented GPU,
2. prepare a reproducible environment,
3. run stock 160M and BitNet 160M with matched settings,
4. test whether validation/checkpointing can be enabled,
5. collect logs/artifacts,
6. terminate the GPU,
7. write down what happened and why.

The learning goal is to move from “does it run for 3 steps?” to “can we operate this as an experiment with logs, checkpoints, validation, and stock-vs-BitNet comparison?”

## Cost guardrail

Use one Lambda `gpu_1x_a10` instance and terminate it when the loop is done or when a blocking failure appears. Lambda listed this instance type at about `$1.29/hour` at launch time.

## Planned configs

Both runs use the same model shape and training knobs except for the BitNet converter.

Common overrides:

```bash
--training.steps 100
--training.local-batch-size 1
--training.seq-len 128
--activation-checkpoint.mode none
--parallelism.data-parallel-shard-degree 1
```

Initial checkpoint/validation probe:

```bash
--checkpoint.enable
--checkpoint.interval 50
--checkpoint.keep-latest-k 2
--validator.freq 50
--validator.steps 2
```

If checkpointing or validation fails for infrastructure reasons, fall back to train-loop-only 100-step runs and document the exact failure.

## Planned commands

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

Status: **running / pending**.

| Run | Steps requested | Completed? | Final train loss | Validation? | Checkpoint? | Notes |
| --- | ---: | --- | ---: | --- | --- | --- |
| stock `llama3_160m` | 100 | pending | pending | pending | pending | pending |
| BitNet `llama3_160m_bitnet` | 100 | pending | pending | pending | pending | pending |

## Raw artifacts

Expected local artifact folder after the run:

```text
docs/bitnet-160m-mvp/experiments/2026-06-08-100-step-shakedown/artifacts/
```

Expected artifacts:

- `environment.txt`
- `stock_100.log`
- `bitnet_100.log`
- checkpoint directory listing if checkpointing succeeds
- parsed loss table if parsing is reliable

## Interpretation

Pending.

## Follow-up questions

- Did validation work for stock and BitNet?
- Did checkpoint save work for stock and BitNet?
- Is BitNet loss decreasing over 100 steps?
- How different are stock and BitNet step times?
- Does memory stay stable?
- Does the converter count mismatch matter operationally?
