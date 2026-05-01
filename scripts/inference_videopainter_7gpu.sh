#!/bin/bash
# VideoPainter (CogVideoX-5B-I2V + branch + LoRA) on inference_160, 7 GPUs.
# Per-sample: edit.py takes one row of the CSV at a time.
# Bypasses LLM + FluxFill via --prebuilt_first_frame_path.
set -u

CONDA_ENV="/home/xinghao/miniconda3/envs/videopainter"
PY="${CONDA_ENV}/bin/python"

VP_HF=$(ls -d ~/.cache/huggingface/hub/models--TencentARC--VideoPainter/snapshots/*/ | head -1)
COG_HF=$(ls -d ~/.cache/huggingface/hub/models--THUDM--CogVideoX-5b-I2V/snapshots/*/ | head -1)

INPAINTING_BRANCH="${VP_HF}VideoPainter/checkpoints/branch"
ID_LORA_PATH="${VP_HF}VideoPainterID/checkpoints"
MODEL_PATH="${COG_HF%/}"

CSV="outputs/videopainter_work/videopainter_inputs.csv"
OUT="outputs/baseline_videopainter_inference_160"
LOGDIR="logs/videopainter_inference_160"

mkdir -p "$OUT" "$LOGDIR"

if [ ! -f "$CSV" ]; then
  echo "ERROR: CSV not found ($CSV). Run scripts/videopainter_prepare.py first." >&2; exit 1
fi
N=$(($(wc -l < "$CSV") - 1))
echo "=========================================="
echo "VideoPainter on inference_160 — 7 GPU shards, N=$N samples"
echo "  branch: $INPAINTING_BRANCH"
echo "  lora:   $ID_LORA_PATH"
echo "  cog:    $MODEL_PATH"
echo "  out:    $OUT"
echo "=========================================="

EDIT_PY="baselines/VideoPainter/infer/edit.py"

run_shard() {
  local shard=$1
  local gpu=$2
  local shardlog="$LOGDIR/shard${shard}_gpu${gpu}"
  mkdir -p "$shardlog"

  for ((i=shard; i<N; i+=7)); do
    sid=$(printf "%03d" $i)
    sample_log="$shardlog/sample_${sid}.log"
    # Pull per-sample fields from CSV via python
    info=$("$PY" -c "
import csv
r = list(csv.DictReader(open('$CSV')))[$i]
print(r['id'], r['first_frame_path'], r['caption'], sep='|')
" 2>/dev/null)
    rid=$(echo "$info" | cut -d'|' -f1)
    ff=$(echo "$info" | cut -d'|' -f2)
    prompt=$(echo "$info" | cut -d'|' -f3)
    out_mp4="$OUT/${rid}.mp4"
    if [ -f "$out_mp4" ]; then
      echo "[shard $shard GPU $gpu sample $sid id=$rid] SKIP (exists)" >> "$sample_log"
      continue
    fi
    echo "[shard $shard GPU $gpu sample $sid id=$rid] running" >> "$sample_log"

    CUDA_VISIBLE_DEVICES=$gpu \
      "$PY" "$EDIT_PY" \
        --prompt "$prompt" \
        --model_path "$MODEL_PATH" \
        --inpainting_branch "$INPAINTING_BRANCH" \
        --id_pool_resample_learnable_path "$ID_LORA_PATH" \
        --output_path "$out_mp4" \
        --num_inference_steps 50 \
        --guidance_scale 6.0 \
        --num_videos_per_prompt 1 \
        --dtype "bfloat16" \
        --generate_type "i2v_inpainting" \
        --inpainting_mask_meta "$CSV" \
        --inpainting_sample_id $i \
        --inpainting_frames 49 \
        --image_or_video_path "" \
        --first_frame_gt \
        --replace_gt \
        --mask_add \
        --down_sample_fps 8 \
        --prev_clip_weight 0.0 \
        --video_editing_instruction "scene-text-edit" \
        --llm_model "None" \
        --dilate_size 0 \
        --lora_rank 256 \
        --prebuilt_first_frame_path "$ff" \
        >> "$sample_log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$out_mp4" ]; then
      echo "[shard $shard GPU $gpu sample $sid id=$rid] DONE -> $out_mp4" >> "$sample_log"
    else
      echo "[shard $shard GPU $gpu sample $sid id=$rid] FAILED rc=$rc" >> "$sample_log"
    fi
  done
}

GPUS=(1 2 3 4 5 6 7)
pids=()
for shard in "${!GPUS[@]}"; do
  gpu=${GPUS[$shard]}
  run_shard $shard $gpu &
  pids+=($!)
  echo "  forked shard=$shard gpu=$gpu pid=${pids[-1]}"
done
for pid in "${pids[@]}"; do wait $pid; done
echo "All VideoPainter shards complete. Outputs: $OUT"
