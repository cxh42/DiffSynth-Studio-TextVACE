"""
TextVACE v3 Batch Inference
============================
Run inference on training samples and unseen videos using the v3 (TargetTextEncoder) model.

Usage:
  python scripts/inference_v3_batch.py \
      --checkpoint models/train/TextVACE_v3_sft/epoch-2.safetensors
"""

import argparse
import json
import os
import sys
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B/"
    "snapshots/574e6a744642ce3bee319afc31496b88bde8aac4"
)


def load_pipeline(checkpoint_path, device="cuda"):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(os.path.join(MODEL_DIR, "diffusion_pytorch_model.safetensors")),
            ModelConfig(os.path.join(MODEL_DIR, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(os.path.join(MODEL_DIR, "Wan2.1_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
    )
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = load_state_dict(checkpoint_path)
    pipe.vace.load_state_dict(state_dict)
    print(f"  use_target_text_encoder: {pipe.vace.use_target_text_encoder}")
    print(f"  has TargetTextEncoder: {hasattr(pipe.vace, 'target_text_encoder')}")
    return pipe


def save_comparison(orig_path, gen_frames, out_path, height=480, width=832):
    """Save side-by-side first frame comparison."""
    orig = VideoData(orig_path, height=height, width=width)
    orig_frame = orig[0]
    gen_frame = gen_frames[0]
    w, h = orig_frame.size
    comp = Image.new("RGB", (w * 2, h))
    comp.paste(orig_frame, (0, 0))
    comp.paste(gen_frame, (w, 0))
    comp.save(out_path)


def run_one(pipe, video_path, mask_path, target_text, prompt, output_path,
            num_frames=17, height=480, width=832, seed=42, steps=50, cfg=5.0):
    vd = VideoData(video_path, height=height, width=width)
    vace_video = [vd[i] for i in range(min(num_frames, len(vd)))]

    md = VideoData(mask_path, height=height, width=width)
    vace_mask = [md[i] for i in range(min(num_frames, len(md)))]

    video = pipe(
        prompt=prompt,
        negative_prompt="",
        vace_video=vace_video,
        vace_video_mask=vace_mask,
        target_text=target_text,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
        num_inference_steps=steps,
        cfg_scale=cfg,
        tiled=True,
    )
    save_video(video, output_path, fps=8, quality=5)
    return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/train/TextVACE_v3_sft/epoch-2.safetensors")
    parser.add_argument("--output_dir", default="outputs/textvace_v3_inference")
    parser.add_argument("--num_train_samples", type=int, default=10)
    parser.add_argument("--num_unseen_samples", type=int, default=20)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    args = parser.parse_args()

    train_dir = os.path.join(args.output_dir, "train_samples")
    unseen_dir = os.path.join(args.output_dir, "unseen_videos")
    for d in [train_dir, os.path.join(train_dir, "frames"),
              unseen_dir, os.path.join(unseen_dir, "frames")]:
        os.makedirs(d, exist_ok=True)

    pipe = load_pipeline(args.checkpoint)

    # ---- Part 1: Training samples ----
    with open("data/processed/parsed_records.json") as f:
        train_records = json.load(f)

    # Select diverse samples
    step = max(1, len(train_records) // args.num_train_samples)
    selected_train = train_records[::step][:args.num_train_samples]

    print(f"\n{'='*60}")
    print(f"Part 1: Training samples ({len(selected_train)} samples)")
    print(f"{'='*60}")

    for i, rec in enumerate(selected_train):
        vid_id = rec["id"]
        orig_path = os.path.join("data/raw", rec["original_video"])
        mask_path = os.path.join("data/raw", rec["mask_video"])

        if not os.path.exists(orig_path) or not os.path.exists(mask_path):
            print(f"  [{i+1}] SKIP {vid_id}: missing files")
            continue

        print(f"  [{i+1}/{len(selected_train)}] {vid_id}: "
              f"\"{rec['source_text']}\" -> \"{rec['target_text']}\"")

        out_path = os.path.join(train_dir, f"{vid_id}_generated.mp4")
        video = run_one(
            pipe, orig_path, mask_path,
            target_text=rec["target_text"],
            prompt=rec["instruction"],
            output_path=out_path,
            num_frames=args.num_frames, seed=args.seed,
            steps=args.steps, cfg=args.cfg_scale,
        )
        save_comparison(orig_path, video,
                        os.path.join(train_dir, "frames", f"{vid_id}_comparison.png"))

    # ---- Part 2: Unseen videos ----
    with open("data/inference_processed/inference_records.json") as f:
        infer_records = json.load(f)

    selected_unseen = infer_records[:args.num_unseen_samples]

    print(f"\n{'='*60}")
    print(f"Part 2: Unseen videos ({len(selected_unseen)} samples)")
    print(f"{'='*60}")

    for i, rec in enumerate(selected_unseen):
        vid_id = rec["id"]
        orig_path = rec["video_path"]
        mask_path = rec["mask_path"]

        if not os.path.exists(orig_path) or not os.path.exists(mask_path):
            print(f"  [{i+1}] SKIP {vid_id}: missing files")
            continue

        target_text = rec["replacement_text"]
        prompt = rec["prompt"]

        print(f"  [{i+1}/{len(selected_unseen)}] {vid_id}: "
              f"\"{rec['original_text']}\" -> \"{target_text}\"")

        out_path = os.path.join(unseen_dir, f"{vid_id}_generated.mp4")
        video = run_one(
            pipe, orig_path, mask_path,
            target_text=target_text,
            prompt=prompt,
            output_path=out_path,
            num_frames=args.num_frames, seed=args.seed,
            steps=args.steps, cfg=args.cfg_scale,
        )
        save_comparison(orig_path, video,
                        os.path.join(unseen_dir, "frames", f"{vid_id}_comparison.png"))

    print(f"\n{'='*60}")
    print(f"All done!")
    print(f"  Training samples: {train_dir}/")
    print(f"  Unseen videos: {unseen_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
