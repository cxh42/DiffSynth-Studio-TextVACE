#!/bin/bash
# TextVACE v2 (Glyph) - 14B Model Training
# 8x H100 80GB, DeepSpeed ZeRO-2, 480P, 49 frames

set -e

export PYTHONUNBUFFERED=1

# Clean pycache to ensure latest code
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/ \
  --dataset_metadata_path data/metadata.csv \
  --data_file_keys "video,vace_video,vace_video_mask,glyph_video" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
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
