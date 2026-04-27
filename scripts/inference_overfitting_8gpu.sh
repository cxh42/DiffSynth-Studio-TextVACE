#!/bin/bash
# Overfitting check: 30-sample subset, 4 ckpts, 8 GPUs each.
# Iterates ckpts sequentially (8 GPUs in parallel per ckpt).
set -e

CONDA_ENV="/home/xinghao/miniconda3/envs/DiffSynth-Studio"
PY="${CONDA_ENV}/bin/python"
export PATH="${CONDA_ENV}/bin:${PATH}"

DATA_DIR="data/inference_new/inference_data"
SUBSET_META="${DATA_DIR}/metadata_overfitting_subset.csv"
OUT_BASE="outputs/inference_overfitting"
LOGDIR="logs/inference_overfitting"
CKPT_DIR="models/train/TextVACE_14B_sft_121f"
CKPT_STEPS=(100 288 432 576)

mkdir -p "$OUT_BASE" "$LOGDIR"

# Build subset CSV if missing
if [ ! -f "$SUBSET_META" ]; then
  echo "Building 30-sample overfitting subset..."
  "$PY" scripts/select_overfitting_subset.py
fi

for step in "${CKPT_STEPS[@]}"; do
  CKPT="${CKPT_DIR}/step-${step}.safetensors"
  OUT="${OUT_BASE}/step-${step}"
  STEP_LOGDIR="${LOGDIR}/step-${step}"
  mkdir -p "$OUT" "$STEP_LOGDIR"

  if [ ! -f "$CKPT" ]; then
    echo "WARN: skipping missing ckpt $CKPT"
    continue
  fi

  echo "=========================================="
  echo "Overfitting eval: step-${step}"
  echo "  ckpt=$CKPT"
  echo "  out=$OUT"
  echo "=========================================="

  pids=()
  for rank in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$rank \
      "$PY" scripts/inference_textvace_14b.py \
        --data_dir "$DATA_DIR" \
        --metadata "$SUBSET_META" \
        --checkpoint "$CKPT" \
        --output_dir "$OUT" \
        --worker_rank $rank \
        --num_workers 8 \
        > "${STEP_LOGDIR}/rank${rank}.log" 2>&1 &
    pids+=($!)
  done

  fail=0
  for pid in "${pids[@]}"; do
    if ! wait $pid; then
      fail=$((fail+1))
    fi
  done

  if [ $fail -eq 0 ]; then
    echo "  step-${step} done. Outputs: $OUT"
  else
    echo "  step-${step} WARN: $fail/8 workers failed. Check ${STEP_LOGDIR}"
  fi
done

echo "=========================================="
echo "Overfitting eval complete. Compare side-by-side:"
for step in "${CKPT_STEPS[@]}"; do
  echo "  ${OUT_BASE}/step-${step}/"
done
echo "Subset metadata: $SUBSET_META"
echo "Subset ids:      ${OUT_BASE}/subset_ids.json"
echo "=========================================="
