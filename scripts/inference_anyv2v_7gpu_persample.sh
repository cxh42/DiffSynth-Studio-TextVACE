#!/bin/bash
# Per-sample AnyV2V driver: each sample → invert → edit → mp4 output
# Shows incremental progress (one mp4 per sample as it finishes)
set -u

CONDA_ENV="/home/xinghao/miniconda3/envs/anyv2v-i2vgen-xl"
PY="${CONDA_ENV}/bin/python"
ANYV2V_DIR="/home/xinghao/DiffSynth-Studio-TextVACE/baselines/AnyV2V/i2vgen-xl"
LOGDIR="${LOGDIR:-logs/anyv2v_inference_160}"
WORK_RESULTS_DIR="${WORK_RESULTS_DIR:-outputs/anyv2v_work/results}"
FLAT_OUT="${FLAT_OUT:-outputs/baseline_anyv2v_inference_160}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS=(1 2 3 4 5 6 7)

mkdir -p "$LOGDIR"

run_shard() {
  local rank=$1
  local gpu=$2
  local shardlog="$LOGDIR/shard${rank}_gpu${gpu}"
  mkdir -p "$shardlog"

  # count per-sample configs for this shard
  local samples=$(ls "$ANYV2V_DIR/configs/group_ddim_inversion/shard${rank}_s"*.json 2>/dev/null | wc -l)
  echo "[shard $rank GPU $gpu] $samples samples to process"

  for i in $(seq 0 $((samples-1))); do
    sid=$(printf "%02d" $i)
    sample_log="$shardlog/sample_${sid}.log"
    inv_cfg="$ANYV2V_DIR/configs/group_ddim_inversion/shard${rank}_s${sid}.json"
    pnp_cfg="$ANYV2V_DIR/configs/group_pnp_edit/shard${rank}_s${sid}.json"
    inv_yaml="$ANYV2V_DIR/configs/group_ddim_inversion/shard${rank}.yaml"
    pnp_yaml="$ANYV2V_DIR/configs/group_pnp_edit/shard${rank}.yaml"

    echo "[shard $rank GPU $gpu sample $sid] inversion" >> "$sample_log"
    CUDA_VISIBLE_DEVICES=$gpu \
      "$PY" "$ANYV2V_DIR/run_group_ddim_inversion.py" \
        --template_config "$inv_yaml" --configs_json "$inv_cfg" \
        >> "$sample_log" 2>&1
    if [ $? -ne 0 ]; then
      echo "[shard $rank GPU $gpu sample $sid] inversion FAILED" >> "$sample_log"
      continue
    fi

    echo "[shard $rank GPU $gpu sample $sid] pnp_edit" >> "$sample_log"
    CUDA_VISIBLE_DEVICES=$gpu \
      "$PY" "$ANYV2V_DIR/run_group_pnp_edit.py" \
        --template_config "$pnp_yaml" --configs_json "$pnp_cfg" \
        >> "$sample_log" 2>&1
    if [ $? -ne 0 ]; then
      echo "[shard $rank GPU $gpu sample $sid] pnp_edit FAILED" >> "$sample_log"
      continue
    fi

    # Postprocess: copy this sample's mp4 to flat output
    vid_name=$(grep '"video_name"' "$inv_cfg" | head -1 | sed -E 's/.*"video_name":\s*"([^"]+)".*/\1/')
    out_dir="${WORK_RESULTS_DIR}/${vid_name}"
    mkdir -p "$FLAT_OUT"
    found_mp4=$(find "$out_dir" -name "*.mp4" 2>/dev/null | head -1)
    if [ -n "$found_mp4" ]; then
      cp "$found_mp4" "$FLAT_OUT/${vid_name}.mp4"
      echo "[shard $rank GPU $gpu sample $sid] DONE -> $FLAT_OUT/${vid_name}.mp4" | tee -a "$sample_log"
    else
      echo "[shard $rank GPU $gpu sample $sid] WARN: no mp4 found in $out_dir" | tee -a "$sample_log"
    fi
  done
  echo "[shard $rank GPU $gpu] all done"
}

mkdir -p "$FLAT_OUT"
echo "=========================================="
echo "AnyV2V per-sample driver — 7 shards"
echo "  GPUs: ${GPUS[*]}"
echo "  output (flat): $FLAT_OUT/"
echo "  work results:  $WORK_RESULTS_DIR/"
echo "=========================================="

pids=()
for rank in "${!GPUS[@]}"; do
  gpu=${GPUS[$rank]}
  run_shard $rank $gpu &
  pids+=($!)
  echo "  forked rank=$rank gpu=$gpu pid=${pids[-1]}"
done

for pid in "${pids[@]}"; do wait $pid; done
echo "=========================================="
echo "All shards complete. Outputs: outputs/baseline_anyv2v_inference_160/"
echo "=========================================="
