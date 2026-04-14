#!/bin/bash
# TextVACE SFT Training Script
# Trains VACE module with glyph condition on Wan2.1-VACE-1.3B
# Expected VRAM: ~27GB with gradient checkpointing offload on RTX 5090 (32GB)
#
# Usage:
#   conda activate DiffSynth-Studio
#   bash scripts/train_textvace.sh 2>&1 | tee logs/train_textvace.log
#
# Or run in background:
#   nohup bash scripts/train_textvace.sh > logs/train_textvace.log 2>&1 &
#   tail -f logs/train_textvace.log   # monitor progress
#   tail -f models/train/TextVACE_sft/training_log.txt  # monitor loss

MODEL_DIR="/home/xinghao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B/snapshots/574e6a744642ce3bee319afc31496b88bde8aac4"

mkdir -p logs

export PYTHONUNBUFFERED=1

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/ \
  --dataset_metadata_path data/metadata.csv \
  --data_file_keys "video,vace_video,vace_video_mask,glyph_video" \
  --height 480 \
  --width 832 \
  --num_frames 17 \
  --dataset_repeat 10 \
  --model_paths "[\"${MODEL_DIR}/diffusion_pytorch_model.safetensors\",\"${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",\"${MODEL_DIR}/Wan2.1_VAE.pth\"]" \
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/TextVACE_sft" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_video_mask,glyph_video" \
  --use_gradient_checkpointing_offload \
  --initialize_model_on_cpu
