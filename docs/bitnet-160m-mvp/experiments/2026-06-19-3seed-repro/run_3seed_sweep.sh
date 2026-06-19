#!/usr/bin/env bash
# LOCKED 3-seed sweep launcher — BitNet 160M reproduction (2026-06-19).
# 6 sequential 1700-step runs: {stock, bitnet} x seeds {42,43,44}.
# Byte-identical training flags to the single-seed run (commit 1559a53ce);
# only --config, --debug.seed, and --dump-folder vary.
#
# Run from the torchtitan repo root, inside the venv with cu128 nightly torch:
#   bash docs/bitnet-160m-mvp/experiments/2026-06-19-3seed-repro/run_3seed_sweep.sh
# Intended to be launched DETACHED via `lambda run_bg` so it survives SSH close.
#
# Per run: stdout tee'd to results/<label>_train.log; filtered curves
# (train.loss/eval.loss/eval.perplexity) to results/<label>_curves.jsonl.
# NOT `set -e`: a failed run is recorded and the remaining runs still proceed.

set -u
cd "$(dirname "$0")/../../../.."   # repo root
EXP=docs/bitnet-160m-mvp/experiments/2026-06-19-3seed-repro
RES="$EXP/results"
mkdir -p "$RES"

CONFIGS=(llama3_160m llama3_160m_bitnet)
SEEDS=(42 43 44)

{
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
  echo "torch: $(python -c 'import torch; print(torch.__version__)')"
  echo "python: $(python --version 2>&1)"
  echo "commit: $(git rev-parse HEAD)"
  echo "started: $(date -u +%FT%TZ)"
} > "$RES/environment.txt"

N=0
for CONFIG in "${CONFIGS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    N=$((N+1))
    LABEL="${CONFIG#llama3_160m}"; LABEL="${LABEL#_}"; [ -z "$LABEL" ] && LABEL="stock"
    LABEL="${LABEL}_s${SEED}"
    OUT="./outputs/${CONFIG}_s${SEED}"
    echo "=== [$(date -u +%FT%TZ)] RUN ${N}/6: ${LABEL} (${CONFIG}, seed ${SEED}) ==="
    rm -rf "$OUT"

    torchrun --nproc_per_node=1 -m torchtitan.train \
      --module torchtitan.models.llama3 --config "$CONFIG" \
      --debug.seed "$SEED" \
      --training.steps 1700 --training.local-batch-size 16 --training.seq-len 2048 \
      --activation-checkpoint.mode selective --parallelism.data-parallel-shard-degree 1 \
      --metrics.log-freq 10 --validator.enable --validator.freq 170 --validator.steps 20 \
      --checkpoint.enable --checkpoint.interval 850 --checkpoint.keep-latest-k 2 \
      --dataloader.dataset c4 --dump-folder "$OUT" \
      2>&1 | tee "$RES/${LABEL}_train.log"
    RC=${PIPESTATUS[0]}
    echo "RUN ${LABEL} exit code: $RC" | tee -a "$RES/run_status.txt"

    grep -hE '"event_name": "(train.loss|eval.loss|eval.perplexity)"' \
      "$OUT/structured_logs/"*.jsonl > "$RES/${LABEL}_curves.jsonl" 2>>"$RES/run_status.txt"
    echo "RUN ${LABEL} curve lines: $(wc -l < "$RES/${LABEL}_curves.jsonl" 2>/dev/null || echo 0)" \
      | tee -a "$RES/run_status.txt"
  done
done

echo "finished: $(date -u +%FT%TZ)" >> "$RES/environment.txt"
echo "ALL_RUNS_DONE $(date -u +%FT%TZ)" > "$RES/SWEEP_DONE.txt"
echo "=== ALL 6 RUNS DONE [$(date -u +%FT%TZ)] ==="
