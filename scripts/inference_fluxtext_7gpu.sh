#!/bin/bash
# Launch FluxText per-frame inference across GPU 1-7.
set -u

PY="/home/xinghao/miniconda3/envs/fluxtext/bin/python"
SCRIPT="scripts/inference_fluxtext_perframe.py"
LOGDIR="logs/fluxtext_inference_160"
GPUS=(1 2 3 4 5 6 7)

mkdir -p "$LOGDIR"

echo "=========================================="
echo "FluxText per-frame on inference_160 — 7 GPUs"
echo "  out: outputs/baseline_fluxtext_inference_160/"
echo "  logs: $LOGDIR"
echo "=========================================="

pids=()
for shard in "${!GPUS[@]}"; do
  gpu=${GPUS[$shard]}
  CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" "$SCRIPT" --shard $shard --num_shards 7 --gpu 0 \
      > "$LOGDIR/shard${shard}_gpu${gpu}.log" 2>&1 &
  pids+=($!)
  echo "  shard=$shard gpu=$gpu pid=${pids[-1]}"
done
for pid in "${pids[@]}"; do wait $pid; done
echo "All FluxText shards complete."
