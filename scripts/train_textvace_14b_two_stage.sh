#!/bin/bash
# TextVACE 14B - Two-Stage Training
# Stage 1: 720P 49 frames, 5 epochs (fast, no CPU offload)
# Stage 2: 720P 121 frames, 1 epoch (from Stage 1 checkpoint, with CPU offload)

set -e

CONDA_ENV="/home/xinghao/miniconda3/envs/DiffSynth-Studio"
export PATH="${CONDA_ENV}/bin:${PATH}"

export PYTHONUNBUFFERED=1
export DIFFSYNTH_MODEL_BASE_PATH="/home/xinghao/DiffSynth-Studio-TextVACE/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_DEEPSPEED_ZERO3_INIT=false

find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

STAGE1_OUTPUT="./models/train/TextVACE_14B_sft_49f"
STAGE2_OUTPUT="./models/train/TextVACE_14B_sft_121f"

# ============================================================
# Stage 1: 720P 49 frames, 5 epochs (ZeRO-3, no CPU offload)
# ============================================================
echo "=========================================="
echo "Stage 1: 720P 49 frames, 5 epochs"
echo "=========================================="

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B_fast.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/ \
  --dataset_metadata_path data/metadata.csv \
  --data_file_keys "video,vace_video,vace_video_mask,glyph_video" \
  --height 720 \
  --width 1280 \
  --num_frames 49 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:Wan2.1_VAE.pth" \
  --tokenizer_path "models/Wan-AI/Wan2.1-VACE-14B/google/umt5-xxl" \
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "${STAGE1_OUTPUT}" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_video_mask,glyph_video" \
  --use_gradient_checkpointing

echo "Stage 1 complete. Checkpoint: ${STAGE1_OUTPUT}/epoch-4.safetensors"

# ============================================================
# Stage 2: 720P 121 frames, 1 epoch (from Stage 1 checkpoint)
# ============================================================
echo "=========================================="
echo "Stage 2: 720P 121 frames, 1 epoch"
echo "=========================================="

# Find the latest checkpoint from Stage 1
STAGE1_CKPT="${STAGE1_OUTPUT}/epoch-4.safetensors"
if [ ! -f "${STAGE1_CKPT}" ]; then
  echo "ERROR: Stage 1 checkpoint not found at ${STAGE1_CKPT}"
  exit 1
fi

# Clean pycache again
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/ \
  --dataset_metadata_path data/metadata.csv \
  --data_file_keys "video,vace_video,vace_video_mask,glyph_video" \
  --height 720 \
  --width 1280 \
  --num_frames 121 \
  --dataset_repeat 10 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-14B:Wan2.1_VAE.pth" \
  --tokenizer_path "models/Wan-AI/Wan2.1-VACE-14B/google/umt5-xxl" \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "${STAGE2_OUTPUT}" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_video_mask,glyph_video" \
  --use_gradient_checkpointing_offload \
  --model_checkpoint_path "${STAGE1_CKPT}"

echo "=========================================="
echo "Training complete!"
echo "Stage 1 output: ${STAGE1_OUTPUT}/"
echo "Stage 2 output: ${STAGE2_OUTPUT}/"
echo "=========================================="
