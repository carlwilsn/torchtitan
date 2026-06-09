# Results — 1700-step seed-locked stock-vs-BitNet run (2026-06-09)

Artifacts from the first real seed-locked comparison. Full analysis in
[`../learning-docs/12-1700-step-seedlocked-comparison.md`](../learning-docs/12-1700-step-seedlocked-comparison.md).

## Files

| File | What it is |
| --- | --- |
| `stock_curves.jsonl` | Filtered structured-log events (`train.loss`, `eval.loss`, `eval.perplexity`) for stock `llama3_160m`. |
| `bitnet_curves.jsonl` | Same, for `llama3_160m_bitnet`. |
| `results_summary.txt` | Extracted final numbers + the full eval/train curves. |

The curve files are the `grep -E '"event_name": "(train.loss|eval.loss|eval.perplexity)"'`
filter of the full per-rank JSONL (the full logs are ~16 MB each, mostly framework
telemetry; these curves are the experiment signal). Re-extract numbers with
`python3` over these JSONL files (each line is one event with `step`, `value`, `event_name`).

## Run config (byte-identical except CONFIG + dump-folder)

```
--training.steps 1700 --training.local-batch-size 16 --training.seq-len 2048
--activation-checkpoint.mode selective --parallelism.data-parallel-shard-degree 1
--metrics.log-freq 10 --validator.enable --validator.freq 170 --validator.steps 20
--checkpoint.enable --checkpoint.interval 850 --checkpoint.keep-latest-k 2
--dataloader.dataset c4
```

Seed 42 (committed `debug.seed=42`), single Lambda A10, torch `2.12.0.dev20260408+cu128`.
Degree-1 FSDP guard re-applied on the box (env-only, uncommitted).

## Headline numbers

| Quantity | Stock | BitNet | Gap |
| --- | ---: | ---: | ---: |
| Final train loss (step 1700) | 1.4123 | 1.5438 | +0.1314 nats |
| Final val loss (step 1700) | 1.4342 | 1.5576 | +0.1233 nats |
| Final val perplexity | 4.1965 | 4.7473 | +0.55 |
| Median throughput (tps) | 33,111 | 18,697 | 1.77× slower |
| Peak GPU memory | 8.34 GiB | 9.81 GiB | +1.47 GiB |

One seed only — a clean data point, not yet a multi-seed quality claim.
