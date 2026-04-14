"""
TextVACE Inference Script
=========================
Load trained TextVACE checkpoint and run inference on test samples.
Supports Pixel-Anchored Denoising for strict background preservation.

Usage:
  conda run -n DiffSynth-Studio python scripts/inference_textvace.py \
      --checkpoint models/train/TextVACE_sft/epoch-4.safetensors \
      --sample_id 0000007_00000 --anchor
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from einops import rearrange

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
    print(f"Loading TextVACE checkpoint: {checkpoint_path}")
    state_dict = load_state_dict(checkpoint_path)
    pipe.vace.load_state_dict(state_dict)
    print(f"VACE glyph_channels: {pipe.vace.glyph_channels}")
    print(f"VACE Conv3D shape: {pipe.vace.vace_patch_embedding.weight.shape}")
    return pipe


def prepare_anchor_latents(pipe, original_video_frames, mask_frames, height, width, num_frames, seed):
    """Prepare anchor latents for Pixel-Anchored Denoising.

    Encodes the original video into latent space and prepares a downsampled
    mask in latent space. During denoising, non-mask regions of the generated
    latents are replaced with the original video's noised latents.
    """
    pipe.load_models_to_device(["vae"])

    # Encode original video to latents
    orig_tensor = pipe.preprocess_video(original_video_frames).to(dtype=pipe.torch_dtype)
    original_latents = pipe.vae.encode(
        orig_tensor, device=pipe.device, tiled=True
    ).to(dtype=pipe.torch_dtype, device=pipe.device)
    pipe.load_models_to_device([])  # offload VAE after encoding

    # Prepare mask in latent space: downsample from pixel space to latent space
    # Latent spatial: H/8, W/8. Latent temporal: (T+3)//4
    mask_tensor = pipe.preprocess_video(mask_frames, min_value=0, max_value=1)
    # Average pool to latent spatial size
    B, C, T, H, W = original_latents.shape
    mask_down = torch.nn.functional.interpolate(
        mask_tensor[:, 0:1],  # single channel
        size=(T, H, W),
        mode="nearest",
    ).to(dtype=pipe.torch_dtype, device=pipe.device)
    # Binarize: > 0.3 means text region
    mask_latent = (mask_down > 0.3).to(dtype=pipe.torch_dtype, device=pipe.device)

    # Generate noise for anchor blending
    torch.manual_seed(seed)
    anchor_noise = torch.randn_like(original_latents).to(dtype=pipe.torch_dtype, device=pipe.device)

    return original_latents, mask_latent, anchor_noise


def run_inference(
    pipe,
    original_video_path: str,
    mask_video_path: str,
    glyph_video_path: str,
    prompt: str,
    output_path: str,
    num_frames: int = 17,
    height: int = 480,
    width: int = 832,
    seed: int = 42,
    num_inference_steps: int = 50,
    use_anchor: bool = False,
    cfg_scale: float = 5.0,
):
    """Run TextVACE inference on a single sample."""
    vace_video_data = VideoData(original_video_path, height=height, width=width)
    vace_video = [vace_video_data[i] for i in range(min(num_frames, len(vace_video_data)))]

    mask_data = VideoData(mask_video_path, height=height, width=width)
    vace_mask = [mask_data[i] for i in range(min(num_frames, len(mask_data)))]

    glyph_data = VideoData(glyph_video_path, height=height, width=width)
    glyph_video = [glyph_data[i] for i in range(min(num_frames, len(glyph_data)))]

    print(f"  Prompt: {prompt}")
    print(f"  Frames: {len(vace_video)}, Steps: {num_inference_steps}, CFG: {cfg_scale}")
    print(f"  Pixel-Anchored Denoising: {'ON' if use_anchor else 'OFF'}")

    # Prepare anchor latents if enabled
    extra_kwargs = {}
    if use_anchor:
        anchor_latents, anchor_mask, anchor_noise = prepare_anchor_latents(
            pipe, vace_video, vace_mask, height, width, num_frames, seed
        )
        extra_kwargs["anchor_latents"] = anchor_latents
        extra_kwargs["anchor_mask_latent"] = anchor_mask
        extra_kwargs["anchor_noise"] = anchor_noise

    # For pixel-anchored denoising, we inject anchor_* into pipeline via
    # a workaround: temporarily add them to the pipeline's internal state
    if use_anchor:
        # We'll pass them through the inputs_shared dict by monkey-patching
        original_call = pipe.__call__

        def patched_call(**kwargs):
            kwargs.update(extra_kwargs)
            return original_call(**kwargs)

        # Actually, the cleaner way: just add them directly since __call__
        # passes **inputs_shared to model_fn, and our denoising loop checks for them.
        # But __call__ only passes defined parameters...
        # Let's use a different approach: set them as pipe attributes temporarily
        pipe._anchor_latents = anchor_latents
        pipe._anchor_mask_latent = anchor_mask
        pipe._anchor_noise = anchor_noise

    video = pipe(
        prompt=prompt,
        negative_prompt="",
        vace_video=vace_video,
        vace_video_mask=vace_mask,
        glyph_video=glyph_video,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        tiled=True,
    )

    if use_anchor:
        del pipe._anchor_latents, pipe._anchor_mask_latent, pipe._anchor_noise

    save_video(video, output_path, fps=8, quality=5)
    print(f"  Saved: {output_path}")
    return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="models/train/TextVACE_sft/epoch-4.safetensors")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--glyph_dir", type=str, default="data/processed/glyph_videos")
    parser.add_argument("--output_dir", type=str, default="outputs/textvace_inference")
    parser.add_argument("--sample_ids", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--anchor", action="store_true",
                        help="Enable Pixel-Anchored Denoising for background preservation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sample_ids:
        sample_ids = args.sample_ids.split(",")
    else:
        sample_ids = [
            "0000007_00000",
            "0000051_00000",
            "0001273_00000",
        ]

    with open("data/processed/parsed_records.json") as f:
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
        glyph_path = os.path.join(args.glyph_dir, sample_id + ".mp4")

        if not all(os.path.exists(p) for p in [orig_path, mask_path, glyph_path]):
            print(f"  SKIP: missing files")
            continue

        suffix = "_anchored" if args.anchor else "_generated"
        output_path = os.path.join(args.output_dir, f"{sample_id}{suffix}.mp4")

        run_inference(
            pipe,
            original_video_path=orig_path,
            mask_video_path=mask_path,
            glyph_video_path=glyph_path,
            prompt=record["instruction"],
            output_path=output_path,
            num_frames=args.num_frames,
            seed=args.seed,
            num_inference_steps=args.steps,
            use_anchor=args.anchor,
            cfg_scale=args.cfg_scale,
        )

    print(f"\n{'='*60}")
    print(f"All done! Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
