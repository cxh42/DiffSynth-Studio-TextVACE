#!/bin/bash
# Full inference: all 235 samples with step-576 ckpt, sharded across 8 GPUs.
set -e

CONDA_ENV="/home/xinghao/miniconda3/envs/DiffSynth-Studio"
PY="${CONDA_ENV}/bin/python"
export PATH="${CONDA_ENV}/bin:${PATH}"

CKPT="${CKPT:-models/train/TextVACE_14B_sft_121f/step-576.safetensors}"
OUT="${OUT:-outputs/inference_full_step576}"
DATA_DIR="data/inference_new/inference_data"
META="${DATA_DIR}/metadata.csv"
LOGDIR="logs/inference_full"

mkdir -p "$OUT" "$LOGDIR"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: ckpt not found: $CKPT" >&2
  exit 1
fi
if [ ! -f "$META" ]; then
  echo "ERROR: metadata not found: $META" >&2
  exit 1
fi

echo "=========================================="
echo "Full inference: ckpt=$CKPT"
echo "                out=$OUT"
echo "                metadata=$META"
echo "=========================================="

pids=()
for rank in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$rank \
    "$PY" scripts/inference_textvace_14b.py \
      --data_dir "$DATA_DIR" \
      --metadata "$META" \
      --checkpoint "$CKPT" \
      --output_dir "$OUT" \
      --worker_rank $rank \
      --num_workers 8 \
      > "$LOGDIR/rank${rank}.log" 2>&1 &
  pids+=($!)
  echo "  launched rank=$rank pid=${pids[-1]} log=$LOGDIR/rank${rank}.log"
done

# wait for all, track failures
fail=0
for pid in "${pids[@]}"; do
  if ! wait $pid; then
    fail=$((fail+1))
  fi
done

echo "=========================================="
if [ $fail -eq 0 ]; then
  echo "All 8 workers OK. Outputs: $OUT"
else
  echo "WARN: $fail/8 workers exited non-zero. Check $LOGDIR"
fi
echo "=========================================="
