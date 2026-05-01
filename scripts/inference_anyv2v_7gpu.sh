#!/bin/bash
# AnyV2V baseline on inference_160, sharded across GPU 1-7.
# Each shard: DDIM inversion → PnP edit (sequential per sample).
set -u

CONDA_ENV="/home/xinghao/miniconda3/envs/anyv2v-i2vgen-xl"
PY="${CONDA_ENV}/bin/python"
ANYV2V_DIR="/home/xinghao/DiffSynth-Studio-TextVACE/baselines/AnyV2V/i2vgen-xl"
LOGDIR="logs/anyv2v_inference_160"
GPUS=(1 2 3 4 5 6 7)

mkdir -p "$LOGDIR"

if [ ! -x "$PY" ]; then
  echo "ERROR: conda env not ready: $PY" >&2; exit 1
fi

run_shard() {
  local rank=$1
  local gpu=$2
  local shardlog="$LOGDIR/shard${rank}_gpu${gpu}"
  mkdir -p "$shardlog"

  echo "[shard $rank GPU $gpu] starting DDIM inversion"
  CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" "$ANYV2V_DIR/run_group_ddim_inversion.py" \
      --template_config "$ANYV2V_DIR/configs/group_ddim_inversion/shard${rank}.yaml" \
      --configs_json "$ANYV2V_DIR/configs/group_ddim_inversion/shard${rank}.json" \
      > "$shardlog/inversion.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[shard $rank GPU $gpu] inversion FAILED rc=$rc; see $shardlog/inversion.log" >&2
    return $rc
  fi

  echo "[shard $rank GPU $gpu] starting PnP edit"
  CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" "$ANYV2V_DIR/run_group_pnp_edit.py" \
      --template_config "$ANYV2V_DIR/configs/group_pnp_edit/shard${rank}.yaml" \
      --configs_json "$ANYV2V_DIR/configs/group_pnp_edit/shard${rank}.json" \
      > "$shardlog/pnp_edit.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[shard $rank GPU $gpu] pnp_edit FAILED rc=$rc; see $shardlog/pnp_edit.log" >&2
    return $rc
  fi
  echo "[shard $rank GPU $gpu] done"
}

echo "=========================================="
echo "AnyV2V on inference_160 — 7 GPU shards"
echo "  GPUs: ${GPUS[*]} (GPU 0 excluded)"
echo "  logs: $LOGDIR"
echo "=========================================="

pids=()
for rank in "${!GPUS[@]}"; do
  gpu=${GPUS[$rank]}
  run_shard $rank $gpu &
  pids+=($!)
  echo "  forked rank=$rank gpu=$gpu pid=${pids[-1]}"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait $pid; then fail=$((fail+1)); fi
done

echo "=========================================="
if [ $fail -eq 0 ]; then
  echo "All 7 shards OK. Raw outputs in outputs/anyv2v_work/results/"
  echo "Run scripts/anyv2v_postprocess.py to flatten to outputs/baseline_anyv2v_inference_160/"
else
  echo "WARN: $fail/7 shards failed. Check $LOGDIR" >&2
fi
echo "=========================================="
