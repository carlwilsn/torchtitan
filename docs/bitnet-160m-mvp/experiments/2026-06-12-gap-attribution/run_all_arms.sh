#!/usr/bin/env bash
# Gap-attribution launcher: 5 sequential 1700-step arms + diagnostics.
# Run from the torchtitan repo root, inside the venv that has the cu128
# nightly torch (see ../2026-06-11-fingerprint-smoke/artifacts/environment.txt).
#
#   bash docs/bitnet-160m-mvp/experiments/2026-06-12-gap-attribution/run_all_arms.sh
#
# Per arm: stdout tee'd to results/<label>_train.log, filtered curves to
# results/<label>_curves.jsonl (same grep as the MVP run), probe_stats JSON to
# results/<label>_probe.json. After all arms: summarize.py table.
# Not `set -e`: a failed arm is recorded and the remaining arms still run.

set -u
cd "$(dirname "$0")/../../../.."   # repo root
EXP=docs/bitnet-160m-mvp/experiments/2026-06-12-gap-attribution
RES="$EXP/results"
mkdir -p "$RES"

LABELS=(stock bitnet fp16_weights no_actquant structure_only)
CONFIGS=(llama3_160m llama3_160m_bitnet llama3_160m_bitnet_fp16_weights llama3_160m_bitnet_no_actquant llama3_160m_bitnet_structure_only)

{
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
  echo "torch: $(python -c 'import torch; print(torch.__version__)')"
  echo "python: $(python --version 2>&1)"
  echo "commit: $(git rev-parse HEAD)"
  echo "started: $(date -u +%FT%TZ)"
} > "$RES/environment.txt"

for i in "${!LABELS[@]}"; do
  LABEL=${LABELS[$i]}
  CONFIG=${CONFIGS[$i]}
  echo "=== [$(date -u +%FT%TZ)] ARM $((i+1))/5: $LABEL ($CONFIG) ==="
  rm -rf "./outputs/$CONFIG"

  torchrun --nproc_per_node=1 -m torchtitan.train \
    --module torchtitan.models.llama3 --config "$CONFIG" \
    --training.steps 1700 --training.local-batch-size 16 --training.seq-len 2048 \
    --activation-checkpoint.mode selective --parallelism.data-parallel-shard-degree 1 \
    --metrics.log-freq 10 --validator.enable --validator.freq 170 --validator.steps 20 \
    --checkpoint.enable --checkpoint.interval 850 --checkpoint.keep-latest-k 2 \
    --dataloader.dataset c4 --dump-folder "./outputs/$CONFIG" \
    2>&1 | tee "$RES/${LABEL}_train.log"
  TRAIN_RC=${PIPESTATUS[0]}
  echo "ARM $LABEL train exit code: $TRAIN_RC" | tee -a "$RES/run_status.txt"

  # Filtered curves (exact MVP mechanism)
  grep -hE '"event_name": "(train.loss|eval.loss|eval.perplexity)"' \
    "./outputs/$CONFIG/structured_logs/"*.jsonl > "$RES/${LABEL}_curves.jsonl" 2>>"$RES/run_status.txt"

  # Per-layer quantization probe on the final checkpoint
  if [ -d "./outputs/$CONFIG/checkpoint/step-1700" ]; then
    python "$EXP/probe_stats.py" --config "$CONFIG" \
      --checkpoint "./outputs/$CONFIG/checkpoint/step-1700" \
      --out "$RES/${LABEL}_probe.json" \
      > "$RES/${LABEL}_probe.log" 2>&1
    echo "ARM $LABEL probe exit code: $?" | tee -a "$RES/run_status.txt"
  else
    echo "ARM $LABEL: no step-1700 checkpoint found, probe skipped" | tee -a "$RES/run_status.txt"
  fi

  # Running attribution table over everything finished so far
  SPECS=""
  for j in $(seq 0 "$i"); do
    f="$RES/${LABELS[$j]}_curves.jsonl"
    [ -s "$f" ] && SPECS="$SPECS ${LABELS[$j]}=$f"
  done
  # shellcheck disable=SC2086
  python "$EXP/summarize.py" $SPECS --baseline stock --out "$RES/summary.md" \
    | tee "$RES/summary.txt"
done

echo "finished: $(date -u +%FT%TZ)" >> "$RES/environment.txt"
echo "=== ALL ARMS DONE [$(date -u +%FT%TZ)] ==="
