"""
Run TextVACE 14B inference on first 8 training samples with new target texts.
Uses: original_video (as vace_video), text_mask (as vace_video_mask),
      new glyph_video (from new_target_records.json), and new instruction as prompt.
"""

import os
import sys
import json
import csv
import torch
import glob
import numpy as np
from PIL import Image
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core import load_state_dict

# ---- Config ----
DATA_DIR = "data"
MODEL_BASE = "models/Wan-AI/Wan2.1-VACE-14B"
CHECKPOINT = "models/train/TextVACE_14B_sft/epoch-4.safetensors"
TOKENIZER_PATH = "models/Wan-AI/Wan2.1-VACE-14B/google/umt5-xxl"
OUTPUT_DIR = "outputs/textvace_14b_new_targets"

HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 49
NUM_INFERENCE_STEPS = 50
CFG_SCALE = 5.0
SEED = 42
NUM_SAMPLES = 8


def load_video_frames(video_path, target_frames=None, resize=None):
    import cv2
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        if resize:
            img = img.resize((resize[1], resize[0]), Image.LANCZOS)
        all_frames.append(img)
    cap.release()

    if target_frames and len(all_frames) > target_frames:
        indices = np.linspace(0, len(all_frames) - 1, target_frames, dtype=int)
        all_frames = [all_frames[i] for i in indices]

    return all_frames


def save_video(frames, output_path, fps=24):
    import subprocess
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    w, h = frames[0].size
    cmd = [
        ffmpeg, '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(np.array(frame).tobytes())
    proc.stdin.close()
    proc.wait()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--sample_index", type=int, default=None, help="Run only this sample index (0-based)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = f"cuda:{args.gpu}"

    # Load new target records
    with open(os.path.join(DATA_DIR, "processed/new_target_records.json")) as f:
        new_records = json.load(f)
    new_records = new_records[:args.num_samples]
    if args.sample_index is not None:
        new_records = [new_records[args.sample_index]]
    print(f"Will process {len(new_records)} samples with new targets")

    # Load pipeline
    print(f"Loading pipeline on {device}...")
    diffusion_shards = sorted(glob.glob(os.path.join(MODEL_BASE, "diffusion_pytorch_model-*.safetensors")))
    model_configs = [ModelConfig(path=diffusion_shards)] + [
        ModelConfig(path=os.path.join(MODEL_BASE, "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(path=os.path.join(MODEL_BASE, "Wan2.1_VAE.pth")),
    ]
    tokenizer_config = ModelConfig(path=TOKENIZER_PATH)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )

    # Load trained VACE checkpoint
    print(f"Loading VACE checkpoint: {args.checkpoint}")
    vace_state_dict = load_state_dict(args.checkpoint)
    load_result = pipe.vace.load_state_dict(vace_state_dict, strict=False)
    print(f"  Loaded {len(vace_state_dict)} keys, missing: {len(load_result.missing_keys)}, unexpected: {len(load_result.unexpected_keys)}")
    del vace_state_dict

    # Process each sample
    for idx, rec in enumerate(new_records):
        video_id = rec["id"]
        new_target = rec["new_target_text"]
        instruction = rec["instruction"]
        output_path = os.path.join(OUTPUT_DIR, f"{video_id}.mp4")

        if os.path.exists(output_path):
            print(f"[{idx+1}/{len(new_records)}] SKIP {video_id} (exists)")
            continue

        print(f"[{idx+1}/{len(new_records)}] {video_id}: '{rec['source_text']}' -> '{new_target}'")
        print(f"  Instruction: {instruction}")

        try:
            target_size = (HEIGHT, WIDTH)
            vace_video = load_video_frames(
                os.path.join(DATA_DIR, f"raw/original_videos/{video_id}.mp4"),
                target_frames=NUM_FRAMES, resize=target_size)
            vace_mask = load_video_frames(
                os.path.join(DATA_DIR, f"raw/text_masks/{video_id}.mp4"),
                target_frames=NUM_FRAMES, resize=target_size)
            glyph = load_video_frames(
                os.path.join(DATA_DIR, f"processed/glyph_videos_new_targets/{video_id}.mp4"),
                target_frames=NUM_FRAMES, resize=target_size)

            print(f"  Loaded: vace_video={len(vace_video)}f, mask={len(vace_mask)}f, glyph={len(glyph)}f")

            output_frames = pipe(
                prompt=instruction,
                negative_prompt="",
                vace_video=vace_video,
                vace_video_mask=vace_mask,
                glyph_video=glyph,
                seed=SEED,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                cfg_scale=CFG_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                tiled=True,
            )

            save_video(output_frames, output_path)
            print(f"  Saved: {output_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone! Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
