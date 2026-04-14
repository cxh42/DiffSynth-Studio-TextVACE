"""
TextVACE Novel Text Inference
==============================
Test generalization by using training videos/masks but with UNSEEN target text.
If glyph conditioning works, the model should render the new text correctly.
If the model just memorized training pairs, it will fail.

Usage:
  conda run -n DiffSynth-Studio python scripts/inference_novel_text.py \
      --checkpoint models/train/TextVACE_sft/epoch-4.safetensors
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.prepare_textvace_data import (
    get_mask_bbox_per_frame, render_text_on_frame,
    _detect_script, _SCRIPT_FONTS,
)

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B/"
    "snapshots/574e6a744642ce3bee319afc31496b88bde8aac4"
)

# Novel text tests: same video/mask, different target text than training
NOVEL_TESTS = [
    {
        "sample_id": "0000007_00000",
        "original_target": "NAD",       # what training used
        "novel_target": "GDP",           # unseen text
        "prompt": "Change ATP to GDP",
        "desc": "English short (ATP->GDP, trained on ATP->NAD)",
    },
    {
        "sample_id": "0000007_00000",
        "original_target": "NAD",
        "novel_target": "AMP",
        "prompt": "Change ATP to AMP",
        "desc": "English short (ATP->AMP)",
    },
    {
        "sample_id": "0000051_00000",
        "original_target": "CARLIFE",
        "novel_target": "SPEED",
        "prompt": "Change TFLCAR to SPEED",
        "desc": "English brand (TFLCAR->SPEED, trained on TFLCAR->CARLIFE)",
    },
    {
        "sample_id": "0000051_00000",
        "original_target": "CARLIFE",
        "novel_target": "AUTOMAX",
        "prompt": "Change TFLCAR to AUTOMAX",
        "desc": "English long (TFLCAR->AUTOMAX)",
    },
    {
        "sample_id": "0001273_00000",
        "original_target": "善",
        "novel_target": "福",
        "prompt": "Change 慈 to 福",
        "desc": "Chinese char (慈->福, trained on 慈->善)",
    },
    {
        "sample_id": "0001273_00000",
        "original_target": "善",
        "novel_target": "道",
        "prompt": "Change 慈 to 道",
        "desc": "Chinese char (慈->道)",
    },
    {
        "sample_id": "0000061_00000",
        "original_target": "k=3",
        "novel_target": "k=7",
        "prompt": "Change k=2 to k=7",
        "desc": "Math (k=2->k=7, trained on k=2->k=3)",
    },
]


def render_glyph_video_for_text(
    target_text, mask_video_path, orig_video_path, font_info_path, output_path
):
    """Render a glyph video for arbitrary target text."""
    # Get video properties
    cap = cv2.VideoCapture(orig_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Get font
    if os.path.exists(font_info_path):
        with open(font_info_path) as f:
            font_path = json.load(f)["resolved_font_path"]
    else:
        font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

    # Override font for non-Latin text
    script = _detect_script(target_text)
    if script in _SCRIPT_FONTS:
        font_path = _SCRIPT_FONTS[script]

    # Get bboxes
    bboxes = get_mask_bbox_per_frame(mask_video_path, total_frames)

    # Render
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    for bbox in bboxes:
        frame = render_text_on_frame(target_text, frame_h, frame_w, bbox, font_path)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def load_pipeline(checkpoint_path, device="cuda"):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
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
    state_dict = load_state_dict(checkpoint_path)
    pipe.vace.load_state_dict(state_dict)
    print(f"Loaded checkpoint. glyph_channels={pipe.vace.glyph_channels}")
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/train/TextVACE_sft/epoch-4.safetensors")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--output_dir", default="outputs/textvace_inference/novel_text")
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "glyphs"), exist_ok=True)

    pipe = load_pipeline(args.checkpoint)

    for test in NOVEL_TESTS:
        sid = test["sample_id"]
        novel_text = test["novel_target"]
        prompt = test["prompt"]
        tag = f"{sid}_{novel_text.replace(' ', '_')}"

        print(f"\n{'='*60}")
        print(f"Test: {test['desc']}")
        print(f"  Novel target: \"{novel_text}\" (training used \"{test['original_target']}\")")

        orig_path = os.path.join(args.raw_dir, f"original_videos/{sid}.mp4")
        mask_path = os.path.join(args.raw_dir, f"text_masks/{sid}.mp4")
        font_info = f"data/processed/font_info/{sid}.json"

        # Render novel glyph video
        glyph_path = os.path.join(args.output_dir, "glyphs", f"{tag}.mp4")
        render_glyph_video_for_text(novel_text, mask_path, orig_path, font_info, glyph_path)

        # Load data
        vace_video = [VideoData(orig_path, height=480, width=832)[i] for i in range(args.num_frames)]
        vace_mask = [VideoData(mask_path, height=480, width=832)[i] for i in range(args.num_frames)]
        glyph_video = [VideoData(glyph_path, height=480, width=832)[i] for i in range(args.num_frames)]

        print(f"  Running inference (steps={args.steps}, cfg={args.cfg_scale})...")
        video = pipe(
            prompt=prompt,
            negative_prompt="",
            vace_video=vace_video,
            vace_video_mask=vace_mask,
            glyph_video=glyph_video,
            num_frames=args.num_frames,
            height=480, width=832,
            seed=args.seed,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg_scale,
            tiled=True,
        )

        out_path = os.path.join(args.output_dir, "videos", f"{tag}.mp4")
        save_video(video, out_path, fps=8, quality=5)
        print(f"  Saved: {out_path}")

        # Save first frame for quick comparison
        cap = cv2.VideoCapture(out_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_path = os.path.join(args.output_dir, "frames", f"{tag}.png")
            cv2.imwrite(frame_path, frame)

    print(f"\n{'='*60}")
    print(f"All done! Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
