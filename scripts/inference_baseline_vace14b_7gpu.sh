#!/bin/bash
# Baseline: zero-shot Wan2.1-VACE-14B with the Family-B instruction template.
# Sharded across GPUs 1-7 (GPU 0 excluded by user request).
set -e

CONDA_ENV="/home/xinghao/miniconda3/envs/DiffSynth-Studio"
PY="${CONDA_ENV}/bin/python"
export PATH="${CONDA_ENV}/bin:${PATH}"

OUT="${OUT:-outputs/baseline_vace14b_zeroshot}"
DATA_DIR="${DATA_DIR:-data/inference_new/inference_data}"
META="${META:-${DATA_DIR}/metadata.csv}"
LOGDIR="${LOGDIR:-logs/baseline_vace14b_zeroshot}"
GPUS=(1 2 3 4 5 6 7)
TEMPLATE='Change {source_text} to {target_text}; preserve everything else.'

mkdir -p "$OUT" "$LOGDIR"

if [ ! -f "$META" ]; then
  echo "ERROR: metadata not found: $META" >&2; exit 1
fi

echo "=========================================="
echo "Baseline: zero-shot Wan2.1-VACE-14B (Family B)"
echo "  out=$OUT"
echo "  prompt template: $TEMPLATE"
echo "  GPUs: ${GPUS[*]} (GPU 0 excluded)"
echo "=========================================="

pids=()
N=${#GPUS[@]}
for rank in $(seq 0 $((N-1))); do
  gpu=${GPUS[$rank]}
  CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" scripts/inference_textvace_14b.py \
      --data_dir "$DATA_DIR" \
      --metadata "$META" \
      --output_dir "$OUT" \
      --worker_rank $rank \
      --num_workers $N \
      --no_load_vace_ckpt \
      --no_glyph \
      --prompt_template "$TEMPLATE" \
      > "$LOGDIR/rank${rank}_gpu${gpu}.log" 2>&1 &
  pids+=($!)
  echo "  launched rank=$rank gpu=$gpu pid=${pids[-1]}"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait $pid; then fail=$((fail+1)); fi
done

if [ $fail -eq 0 ]; then
  echo "=========================================="
  echo "Baseline done. Outputs: $OUT"
  echo "=========================================="
else
  echo "WARN: $fail/$N workers exited non-zero. Check $LOGDIR" >&2
fi
