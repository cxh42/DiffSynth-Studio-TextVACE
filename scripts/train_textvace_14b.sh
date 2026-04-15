#!/bin/bash
# TextVACE v2 (Glyph) - 14B Model Training
# 8x H100 80GB, DeepSpeed ZeRO-2, 720P, 121 frames

set -e

export PYTHONUNBUFFERED=1
export DIFFSYNTH_MODEL_BASE_PATH="/home/xinghao/DiffSynth-Studio-TextVACE/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_DEEPSPEED_ZERO3_INIT=false

# Clean pycache to ensure latest code
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
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/TextVACE_14B_sft" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_video_mask,glyph_video" \
  --use_gradient_checkpointing
