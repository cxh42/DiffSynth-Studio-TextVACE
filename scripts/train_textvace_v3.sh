#!/bin/bash
# TextVACE v3 Training Script - Target Text Encoder approach
# Uses character-level text encoding instead of glyph video rendering
# Expected VRAM: ~25-27GB on RTX 5090 (32GB)
#
# Usage:
#   conda activate DiffSynth-Studio
#   bash scripts/train_textvace_v3.sh 2>&1 | tee logs/train_textvace_v3.log
#
# Or run in background:
#   nohup bash scripts/train_textvace_v3.sh > logs/train_textvace_v3.log 2>&1 &
#   tail -f logs/train_textvace_v3.log
#   tail -f models/train/TextVACE_v3_sft/training_log.txt

MODEL_DIR="/home/xinghao/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B/snapshots/574e6a744642ce3bee319afc31496b88bde8aac4"

mkdir -p logs

export PYTHONUNBUFFERED=1

ACCELERATE=/home/xinghao/anaconda3/envs/DiffSynth-Studio/bin/accelerate

$ACCELERATE launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/ \
  --dataset_metadata_path data/metadata_v3.csv \
  --data_file_keys "video,vace_video,vace_video_mask" \
  --height 480 \
  --width 832 \
  --num_frames 17 \
  --dataset_repeat 10 \
  --model_paths "[\"${MODEL_DIR}/diffusion_pytorch_model.safetensors\",\"${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",\"${MODEL_DIR}/Wan2.1_VAE.pth\"]" \
  --learning_rate 5e-5 \
  --num_epochs 3 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/TextVACE_v3_sft" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_video_mask,target_text" \
  --dataset_num_workers 4 \
  --initialize_model_on_cpu
