"""
TextVACE inference on unseen videos (not in training data).
Uses pre-processed data from prepare_inference_data.py.
"""

import json
import os
import sys

import cv2
import numpy as np
import torch

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
    print(f"Loaded. glyph_channels={pipe.vace.glyph_channels}")
    return pipe


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/train/TextVACE_sft/epoch-4.safetensors")
    parser.add_argument("--records", default="data/inference_processed/inference_records.json")
    parser.add_argument("--output_dir", default="outputs/textvace_inference/unseen_videos")
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "frames"), exist_ok=True)

    with open(args.records) as f:
        records = json.load(f)

    # Filter out records where replacement == original
    records = [r for r in records if r["original_text"].strip().lower() != r["replacement_text"].strip().lower()]
    records = records[:args.max_samples]
    print(f"Running inference on {len(records)} unseen samples")

    pipe = load_pipeline(args.checkpoint)

    for i, rec in enumerate(records):
        vid_id = rec["id"]
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(records)}] {vid_id}: \"{rec['original_text']}\" -> \"{rec['replacement_text']}\"")

        vace_video = [VideoData(rec["video_path"], height=480, width=832)[j]
                      for j in range(args.num_frames)]
        vace_mask = [VideoData(rec["mask_path"], height=480, width=832)[j]
                     for j in range(args.num_frames)]
        glyph_video = [VideoData(rec["glyph_path"], height=480, width=832)[j]
                       for j in range(args.num_frames)]

        print(f"  Prompt: {rec['prompt']}")
        video = pipe(
            prompt=rec["prompt"],
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

        out_path = os.path.join(args.output_dir, "videos", f"{vid_id}.mp4")
        save_video(video, out_path, fps=8, quality=5)
        print(f"  Saved: {out_path}")

        # Save comparison: original frame | generated frame
        cap = cv2.VideoCapture(out_path)
        _, gen_frame = cap.read()
        cap.release()

        cap2 = cv2.VideoCapture(rec["video_path"])
        _, orig_frame = cap2.read()
        cap2.release()

        if gen_frame is not None and orig_frame is not None:
            orig_frame = cv2.resize(orig_frame, (832, 480))
            gen_frame = cv2.resize(gen_frame, (832, 480))
            cv2.putText(orig_frame, f"Original: {rec['original_text']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(gen_frame, f"Generated: {rec['replacement_text']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            combined = np.hstack([orig_frame, gen_frame])
            cv2.imwrite(os.path.join(args.output_dir, "frames", f"{vid_id}_comparison.png"), combined)

    print(f"\n{'='*60}")
    print(f"All done! Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
