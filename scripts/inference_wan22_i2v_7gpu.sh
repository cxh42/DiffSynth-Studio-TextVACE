#!/bin/bash
# Wan2.2-I2V-A14B baseline (Family D variant): edited first frame + prompt -> video.
# Sharded across GPU 1-7 (GPU 0 excluded).
set -u

CONDA_ENV="/home/xinghao/miniconda3/envs/DiffSynth-Studio"
PY="${CONDA_ENV}/bin/python"
export PATH="${CONDA_ENV}/bin:${PATH}"

OUT="${OUT:-outputs/baseline_wan22_i2v_inference_160}"
LOGDIR="${LOGDIR:-logs/baseline_wan22_i2v_inference_160}"
GPUS=(1 2 3 4 5 6 7)

mkdir -p "$OUT" "$LOGDIR"

echo "=========================================="
echo "Wan2.2-I2V-A14B baseline (Family D variant)"
echo "  out=$OUT"
echo "  GPUs: ${GPUS[*]} (GPU 0 excluded)"
echo "  config: 1280x720, 121 frames, 50 steps, cfg=5.0"
echo "  prompt = metadata.csv 'instruction' column (Family-B template)"
echo "=========================================="

pids=()
N=${#GPUS[@]}
for rank in $(seq 0 $((N-1))); do
  gpu=${GPUS[$rank]}
  CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" scripts/inference_wan22_i2v.py \
      --output_dir "$OUT" \
      --worker_rank $rank \
      --num_workers $N \
      > "$LOGDIR/rank${rank}_gpu${gpu}.log" 2>&1 &
  pids+=($!)
  echo "  launched rank=$rank gpu=$gpu pid=${pids[-1]}"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait $pid; then fail=$((fail+1)); fi
done

echo "=========================================="
if [ $fail -eq 0 ]; then
  echo "All 7 workers OK. Outputs: $OUT"
else
  echo "WARN: $fail/$N workers exited non-zero. Check $LOGDIR" >&2
fi
echo "=========================================="
