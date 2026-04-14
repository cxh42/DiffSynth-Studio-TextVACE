"""
TextVACE v3 Inference Script
=============================
Load trained TextVACE v3 checkpoint (TargetTextEncoder) and run inference.
Uses target_text string instead of glyph video.

Usage:
  conda run -n DiffSynth-Studio python scripts/inference_textvace_v3.py \
      --checkpoint models/train/TextVACE_v3_sft/epoch-3.safetensors \
      --sample_id 0000007_00000
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B/"
    "snapshots/574e6a744642ce3bee319afc31496b88bde8aac4"
)


def load_pipeline(checkpoint_path: str, device: str = "cuda"):
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
    print(f"Loading TextVACE v3 checkpoint: {checkpoint_path}")
    state_dict = load_state_dict(checkpoint_path)
    pipe.vace.load_state_dict(state_dict)
    print(f"VACE use_target_text_encoder: {pipe.vace.use_target_text_encoder}")
    print(f"Has TargetTextEncoder: {hasattr(pipe.vace, 'target_text_encoder')}")
    return pipe


def run_inference(
    pipe,
    original_video_path: str,
    mask_video_path: str,
    target_text: str,
    prompt: str,
    output_path: str,
    num_frames: int = 17,
    height: int = 480,
    width: int = 832,
    seed: int = 42,
    num_inference_steps: int = 50,
    cfg_scale: float = 5.0,
):
    """Run TextVACE v3 inference on a single sample."""
    vace_video_data = VideoData(original_video_path, height=height, width=width)
    vace_video = [vace_video_data[i] for i in range(min(num_frames, len(vace_video_data)))]

    mask_data = VideoData(mask_video_path, height=height, width=width)
    vace_mask = [mask_data[i] for i in range(min(num_frames, len(mask_data)))]

    print(f"  Prompt: {prompt}")
    print(f"  Target text: \"{target_text}\"")
    print(f"  Frames: {len(vace_video)}, Steps: {num_inference_steps}, CFG: {cfg_scale}")

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
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        tiled=True,
    )

    save_video(video, output_path, fps=8, quality=5)
    print(f"  Saved: {output_path}")
    return video


def save_comparison_frame(original_path, generated_frames, output_path, frame_idx=0):
    """Save side-by-side comparison of original and generated first frame."""
    orig_data = VideoData(original_path, height=480, width=832)
    orig_frame = orig_data[frame_idx]
    gen_frame = generated_frames[frame_idx]

    # Side by side
    w, h = orig_frame.size
    comparison = Image.new("RGB", (w * 2, h))
    comparison.paste(orig_frame, (0, 0))
    comparison.paste(gen_frame, (w, 0))
    comparison.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="models/train/TextVACE_v3_sft/epoch-3.safetensors")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="outputs/textvace_v3_inference")
    parser.add_argument("--sample_ids", type=str, default=None)
    parser.add_argument("--records_json", type=str, default="data/processed/parsed_records.json")
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    if args.sample_ids:
        sample_ids = args.sample_ids.split(",")
    else:
        sample_ids = [
            "0000007_00000",
            "0000051_00000",
            "0001273_00000",
        ]

    with open(args.records_json) as f:
        records = {r["id"]: r for r in json.load(f)}

    pipe = load_pipeline(args.checkpoint)

    for sample_id in sample_ids:
        if sample_id not in records:
            print(f"WARNING: {sample_id} not found, skipping")
            continue

        record = records[sample_id]
        print(f"\n{'='*60}")
        print(f"Sample: {sample_id}")
        print(f"  Edit: \"{record['source_text']}\" -> \"{record['target_text']}\"")

        orig_path = os.path.join(args.raw_dir, record["original_video"])
        mask_path = os.path.join(args.raw_dir, record["mask_video"])

        if not all(os.path.exists(p) for p in [orig_path, mask_path]):
            print(f"  SKIP: missing files")
            continue

        output_path = os.path.join(args.output_dir, f"{sample_id}_generated.mp4")

        video = run_inference(
            pipe,
            original_video_path=orig_path,
            mask_video_path=mask_path,
            target_text=record["target_text"],
            prompt=record["instruction"],
            output_path=output_path,
            num_frames=args.num_frames,
            seed=args.seed,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg_scale,
        )

        # Save comparison frame
        comp_path = os.path.join(frames_dir, f"{sample_id}_comparison.png")
        save_comparison_frame(orig_path, video, comp_path)

    print(f"\n{'='*60}")
    print(f"All done! Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
