#!/usr/bin/env bash
# Width-sweep ternary-gap launcher: 4 fresh seed-locked 1700-step runs.
#   50M pair (bs16) + 400M pair (bs16, with bs8 OOM fallback).
# Reuses the already-verified 160M pair (NOT re-run here).
#
# Run order: 50m_stock -> 50m_bitnet -> 400M pair.
# 400M OOM CONTINGENCY: a prior 400M run died ~step 640 on an A10 (24GB). The
# 400M-bitnet arm (heavier: STE master weights + per-layer pre-norms) is run
# FIRST as a CANARY at bs16. If it does not reach "Training completed" (OOM at a
# train OR validation step), BOTH 400M arms are rerun at bs8 so the 400M A/B
# stays internally fair (same batch for both arms). The 50M pair is always bs16.
#
#   tmux session: widthsweep ; log: ~/widthsweep.log
set -u
cd ~/torchtitan
export PATH=$HOME/.local/bin:$PATH
RES=~/widthsweep_results
mkdir -p "$RES"

COMMON="--training.seq-len 2048 --activation-checkpoint.mode selective \
--parallelism.data-parallel-shard-degree 1 --metrics.log-freq 10 \
--validator.enable --validator.freq 170 --validator.steps 20 \
--checkpoint.enable --checkpoint.interval 850 --checkpoint.keep-latest-k 2 \
--dataloader.dataset c4"

{
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
  echo "torch: $(python -c 'import torch; print(torch.__version__)')"
  echo "python: $(python --version 2>&1)"
  echo "commit: $(git rev-parse HEAD)"
  echo "started: $(date -u +%FT%TZ)"
} > "$RES/environment.txt"

# run_one LABEL CONFIG BS STEPS -> returns 0 iff the run reached "Training completed"
run_one () {
  local LABEL=$1 CONFIG=$2 BS=$3 STEPS=$4
  echo "=== [$(date -u +%FT%TZ)] RUN $LABEL ($CONFIG) bs=$BS steps=$STEPS ==="
  rm -rf "./outputs/$LABEL"
  torchrun --nproc_per_node=1 -m torchtitan.train \
    --module torchtitan.models.llama3 --config "$CONFIG" \
    --training.steps "$STEPS" --training.local-batch-size "$BS" $COMMON \
    --dump-folder "./outputs/$LABEL" 2>&1 | tee "$RES/${LABEL}_train.log"
  echo "RUN $LABEL (bs=$BS) pipe-exit ${PIPESTATUS[0]}" | tee -a "$RES/run_status.txt"
  # filtered curves (exact MVP mechanism)
  grep -hE '"event_name": "(train.loss|eval.loss|eval.perplexity)"' \
    "./outputs/$LABEL/structured_logs/"*.jsonl > "$RES/${LABEL}_curves.jsonl" 2>>"$RES/run_status.txt" || true
  if grep -q "Training completed" "$RES/${LABEL}_train.log"; then
    echo "RUN $LABEL OK (completed)" | tee -a "$RES/run_status.txt"; return 0
  else
    echo "RUN $LABEL FAILED (no 'Training completed')" | tee -a "$RES/run_status.txt"; return 1
  fi
}

# ---- 50M pair @ bs16 ----
run_one 50m_stock  llama3_50m        16 1700
run_one 50m_bitnet llama3_50m_bitnet 16 1700

# ---- 400M pair: canary-first (bitnet) @ bs16, bs8 fallback for BOTH ----
BS400=16
if run_one 400m_bitnet llama3_400m_bitnet 16 1700; then
  run_one 400m_stock llama3_400m 16 1700
else
  echo "=== 400M-bitnet OOM/failed at bs16 -> FALLBACK: both 400M arms at bs8 ===" | tee -a "$RES/run_status.txt"
  BS400=8
  run_one 400m_bitnet llama3_400m_bitnet 8 1700
  run_one 400m_stock  llama3_400m        8 1700
fi
echo "BS400_FINAL=$BS400" | tee -a "$RES/run_status.txt"

echo "finished: $(date -u +%FT%TZ)" >> "$RES/environment.txt"
echo "=== ALL RUNS DONE [$(date -u +%FT%TZ)] BS400=$BS400 ==="
