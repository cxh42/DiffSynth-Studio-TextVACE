"""
TextVACE 14B Inference. Reads a metadata CSV, runs pipeline per row.
Supports multi-GPU sharding via --worker_rank / --num_workers (one process per GPU).
"""

import os
import sys
import csv
import torch
import json
from PIL import Image
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# ---- Defaults ----
DATA_DIR_DEFAULT = "data/inference_new/inference_data"
METADATA_DEFAULT = "data/inference_new/inference_data/metadata.csv"
MODEL_BASE = "models/Wan-AI/Wan2.1-VACE-14B"
CHECKPOINT_DEFAULT = "models/train/TextVACE_14B_sft_121f/step-576.safetensors"
TOKENIZER_PATH = "models/Wan-AI/Wan2.1-VACE-14B/google/umt5-xxl"
OUTPUT_DIR_DEFAULT = "outputs/inference_full_step576"

HEIGHT_DEFAULT = 720
WIDTH_DEFAULT = 1280
NUM_FRAMES_DEFAULT = 121
NUM_INFERENCE_STEPS = 50
CFG_SCALE = 5.0
SEED = 42


def load_video_frames(video_path, target_frames=None, resize=None):
    """Load video as list of PIL Images, optionally resized.

    If target_frames is set:
      - Longer videos are uniformly subsampled to target_frames.
      - Shorter videos are right-padded by repeating the last frame
        (matches training-side LoadVideo in diffsynth/core/data/operators.py).
    """
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
    elif target_frames and len(all_frames) < target_frames:
        if not all_frames:
            raise ValueError(f"empty video: {video_path}")
        last = all_frames[-1]
        all_frames.extend([last] * (target_frames - len(all_frames)))

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
    parser.add_argument("--data_dir", type=str, default=DATA_DIR_DEFAULT, help="Base dir; CSV paths resolve relative to this.")
    parser.add_argument("--metadata", type=str, default=METADATA_DEFAULT, help="Path to metadata CSV.")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples (after sharding).")
    parser.add_argument("--gpu", type=int, default=0, help="cuda:N. Usually 0 because CUDA_VISIBLE_DEVICES isolates per-worker.")
    parser.add_argument("--worker_rank", type=int, default=0, help="Worker index in [0, num_workers).")
    parser.add_argument("--num_workers", type=int, default=1, help="Total parallel workers (one process per GPU).")
    parser.add_argument("--height", type=int, default=HEIGHT_DEFAULT)
    parser.add_argument("--width", type=int, default=WIDTH_DEFAULT)
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES_DEFAULT)
    args = parser.parse_args()

    if args.num_workers < 1 or not (0 <= args.worker_rank < args.num_workers):
        raise ValueError(f"bad sharding: rank={args.worker_rank}, num_workers={args.num_workers}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu}"

    print(f"[rank {args.worker_rank}/{args.num_workers}] Loading pipeline on {device}...")

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
    with open(args.metadata) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)

    total = len(metadata)
    # Shard: this worker takes rows where idx % num_workers == worker_rank
    metadata = [row for idx, row in enumerate(metadata) if idx % args.num_workers == args.worker_rank]
    print(f"[rank {args.worker_rank}/{args.num_workers}] {len(metadata)}/{total} samples assigned")

    if args.num_samples:
        metadata = metadata[:args.num_samples]

    for idx, row in enumerate(metadata):
        video_id = row.get("id") or os.path.basename(row["vace_video"]).replace(".mp4", "")
        output_path = os.path.join(args.output_dir, f"{video_id}.mp4")

        if os.path.exists(output_path):
            print(f"[rank {args.worker_rank}] [{idx+1}/{len(metadata)}] SKIP {video_id} (exists)")
            continue

        print(f"[rank {args.worker_rank}] [{idx+1}/{len(metadata)}] Processing {video_id}: {row['prompt']}")

        try:
            # Load input videos, sample to num_frames, resize to target resolution
            target_size = (args.height, args.width)  # (h, w)
            vace_video = load_video_frames(os.path.join(args.data_dir, row["vace_video"]), target_frames=args.num_frames, resize=target_size)
            vace_mask = load_video_frames(os.path.join(args.data_dir, row["vace_video_mask"]), target_frames=args.num_frames, resize=target_size)
            glyph = load_video_frames(os.path.join(args.data_dir, row["glyph_video"]), target_frames=args.num_frames, resize=target_size)

            # Run inference
            output_frames = pipe(
                prompt=row["prompt"],
                negative_prompt="",
                vace_video=vace_video,
                vace_video_mask=vace_mask,
                glyph_video=glyph,
                seed=SEED,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                cfg_scale=CFG_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                tiled=True,
            )

            # Save output
            save_video(output_frames, output_path)
            print(f"[rank {args.worker_rank}]   Saved: {output_path}")

        except Exception as e:
            print(f"[rank {args.worker_rank}]   ERROR on {video_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"[rank {args.worker_rank}/{args.num_workers}] Done. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
