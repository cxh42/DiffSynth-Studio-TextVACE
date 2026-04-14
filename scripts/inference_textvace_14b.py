"""
TextVACE 14B Inference on training data with the trained checkpoint.
Uses original videos + masks + glyph_videos from training data.
"""

import os
import sys
import csv
import torch
import json
from PIL import Image
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# ---- Config ----
DATA_DIR = "data"
MODEL_BASE = "models/Wan-AI/Wan2.1-VACE-14B"
CHECKPOINT = "models/train/TextVACE_14B_sft/epoch-4.safetensors"
TOKENIZER_PATH = "models/Wan-AI/Wan2.1-VACE-14B/google/umt5-xxl"
OUTPUT_DIR = "outputs/textvace_14b_inference"

HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 49
NUM_INFERENCE_STEPS = 50
CFG_SCALE = 5.0
SEED = 42


def load_video_frames(video_path, target_frames=None, resize=None):
    """Load video as list of PIL Images, uniformly sampled to target_frames, optionally resized."""
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
            img = img.resize((resize[1], resize[0]), Image.LANCZOS)  # (width, height)
        all_frames.append(img)
    cap.release()

    if target_frames and len(all_frames) > target_frames:
        import numpy as np
        indices = np.linspace(0, len(all_frames) - 1, target_frames, dtype=int)
        all_frames = [all_frames[i] for i in indices]

    return all_frames


def save_video(frames, output_path, fps=24):
    """Save list of PIL Images as H.264 video."""
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
        import numpy as np
        proc.stdin.write(np.array(frame).tobytes())
    proc.stdin.close()
    proc.wait()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu}"

    print(f"Loading pipeline on {device}...")

    # Model configs: base 14B (without trained checkpoint - load separately)
    import glob
    from diffsynth.core import load_state_dict
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

    # Load trained VACE checkpoint on top of base VACE
    print(f"Loading VACE checkpoint: {args.checkpoint}")
    vace_state_dict = load_state_dict(args.checkpoint)
    load_result = pipe.vace.load_state_dict(vace_state_dict, strict=False)
    print(f"  Loaded {len(vace_state_dict)} keys, missing: {len(load_result.missing_keys)}, unexpected: {len(load_result.unexpected_keys)}")
    if load_result.unexpected_keys:
        print(f"  Unexpected keys: {load_result.unexpected_keys[:5]}...")
    del vace_state_dict

    # Read metadata
    metadata = []
    with open(os.path.join(DATA_DIR, "metadata.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)

    print(f"Loaded {len(metadata)} samples")

    if args.num_samples:
        metadata = metadata[:args.num_samples]

    for idx, row in enumerate(metadata):
        video_id = os.path.basename(row["vace_video"]).replace(".mp4", "")
        output_path = os.path.join(args.output_dir, f"{video_id}.mp4")

        if os.path.exists(output_path):
            print(f"[{idx+1}/{len(metadata)}] SKIP {video_id} (exists)")
            continue

        print(f"[{idx+1}/{len(metadata)}] Processing {video_id}: {row['prompt']}")

        try:
            # Load input videos, sample to NUM_FRAMES, resize to target resolution
            target_size = (HEIGHT, WIDTH)  # (h, w)
            vace_video = load_video_frames(os.path.join(DATA_DIR, row["vace_video"]), target_frames=NUM_FRAMES, resize=target_size)
            vace_mask = load_video_frames(os.path.join(DATA_DIR, row["vace_video_mask"]), target_frames=NUM_FRAMES, resize=target_size)
            glyph = load_video_frames(os.path.join(DATA_DIR, row["glyph_video"]), target_frames=NUM_FRAMES, resize=target_size)

            # Run inference
            output_frames = pipe(
                prompt=row["prompt"],
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

            # Save output
            save_video(output_frames, output_path)
            print(f"  Saved: {output_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone! Results in {args.output_dir}")


if __name__ == "__main__":
    main()
