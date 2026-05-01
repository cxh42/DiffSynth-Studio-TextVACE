"""
FluxText per-frame baseline (Family A).
For each test video: edit every one of its 120 frames independently with FLUX-Text,
then assemble into an mp4 at 1280x720@24fps.

Usage: python inference_fluxtext_perframe.py --shard 0 --num_shards 7 --gpu 1
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
# Compat shim: numpy 2.x removed np.int0; t3_dataset.py uses it.
if not hasattr(np, "int0"):
    np.int0 = np.intp
import torch
from PIL import Image, ImageFont
from safetensors.torch import load_file
import yaml

REPO = Path("/home/xinghao/DiffSynth-Studio-TextVACE")
FT = REPO / "baselines" / "FluxText"
sys.path.insert(0, str(FT))
os.chdir(str(FT))

from src.flux.condition import Condition           # noqa: E402
from src.flux.generate_fill import generate_fill   # noqa: E402
from src.train.model import OminiModelFIll         # noqa: E402
from eval.t3_dataset import draw_glyph2            # noqa: E402

INF = REPO / "data" / "inference_new" / "inference_160"
META = INF / "metadata.csv"
OUT_DIR = REPO / "outputs" / "baseline_fluxtext_inference_160"
LORA_PATH = next((Path.home() / ".cache/huggingface/hub/models--GD-ML--FLUX-Text/snapshots").glob("*"))
LORA_FILE = LORA_PATH / "model_multisize" / "pytorch_lora_weights.safetensors"
CONFIG_FILE = LORA_PATH / "model_multisize" / "config.yaml"
FONT_FILE = FT / "font" / "Arial_Unicode.ttf"

# 720p target (matches our evaluation grid)
TGT_W, TGT_H = 1280, 720
N_FRAMES = 120
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 30.0
SEED = 42


def load_video_frames(path, n=N_FRAMES, size=(TGT_W, TGT_H)):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret: break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    if len(frames) >= n:
        frames = frames[:n]
    else:
        # pad
        frames = frames + [frames[-1]] * (n - len(frames))
    return frames


def load_mask_frames(path, n=N_FRAMES, size=(TGT_W, TGT_H)):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ret, fr = cap.read()
        if not ret: break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        bin_ = (gray > 127).astype(np.uint8) * 255
        bin_ = cv2.resize(bin_, size, interpolation=cv2.INTER_NEAREST)
        out.append(bin_)
    cap.release()
    if len(out) >= n:
        out = out[:n]
    else:
        out = out + [out[-1]] * (n - len(out))
    return out


def write_mp4(frames_rgb, path, fps=24):
    h, w = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for fr in frames_rgb:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()


def render_glyph_for_mask(mask_np, target_text, w, h, font):
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Empty mask: return black (no glyphs)
        return np.zeros((h, w, 3), dtype=np.uint8)
    contour = max(contours, key=cv2.contourArea)
    glyph = draw_glyph2(font, target_text, contour, scale=1, width=w, height=h)
    # draw_glyph2 returns (H, W, 1) float64 in {0,1}; squeeze + scale to uint8
    if glyph.ndim == 3 and glyph.shape[-1] == 1:
        glyph = glyph[..., 0]
    glyph = (glyph * 255).astype(np.uint8) if glyph.max() <= 1 else glyph.astype(np.uint8)
    return glyph


def edit_frame(pipe, model_config, frame_rgb, mask_np, target_text, font, gen):
    img_pil = Image.fromarray(frame_rgb)
    mask_pil = Image.fromarray(mask_np).convert('RGB')
    glyph_np = render_glyph_for_mask(mask_np, target_text, TGT_W, TGT_H, font)
    glyph_pil = Image.fromarray(glyph_np).convert('RGB')

    hint = np.array(mask_pil) / 255.0
    cond = np.array(glyph_pil)
    cond = (255 - cond) / 255.0
    condition_img = [cond, hint, img_pil]
    condition = Condition(condition_type='word_fill',
                          condition=condition_img,
                          position_delta=[0, 0])
    res = generate_fill(
        pipe,
        prompt=f'Text that reads "{target_text}".',
        conditions=[condition],
        height=TGT_H,
        width=TGT_W,
        generator=gen,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        model_config=model_config,
        default_lora=True,
    )
    return np.array(res.images[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num_shards", type=int, default=7)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}"

    # Load model
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)
    print(f"[shard {args.shard}] Loading FLUX.1-Fill-dev + FLUX-Text LoRA on {device}")
    trainable = OminiModelFIll(
        flux_pipe_id=config["flux_path"],
        lora_config=config["train"]["lora_config"],
        device=device,
        dtype=torch.bfloat16,
        optimizer_config=config["train"]["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=config["train"].get("gradient_checkpointing", False),
        byt5_encoder_config=config["train"].get("byt5_encoder", None),
    )
    state_dict = load_file(str(LORA_FILE))
    state_dict1 = {x.replace('lora_A', 'lora_A.default').replace('lora_B', 'lora_B.default').replace('transformer.', ''): v
                   for x, v in state_dict.items()}
    trainable.transformer.load_state_dict(state_dict1, strict=False)
    pipe = trainable.flux_pipe
    pipe.set_progress_bar_config(disable=True)

    font = ImageFont.truetype(str(FONT_FILE), size=60)
    gen = torch.Generator(device=device).manual_seed(SEED)

    # Read metadata + filter shard
    rows = list(csv.DictReader(open(META)))
    rows.sort(key=lambda r: r["id"])
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}] {len(rows)} samples assigned")

    for ri, r in enumerate(rows):
        rid = r["id"]
        out_path = OUT_DIR / f"{rid}.mp4"
        if out_path.exists():
            print(f"[shard {args.shard}] [{ri+1}/{len(rows)}] SKIP {rid}")
            continue
        target = r["target_text"]
        video = INF / r["vace_video"]
        mask  = INF / r["vace_video_mask"]

        print(f"[shard {args.shard}] [{ri+1}/{len(rows)}] {rid}  target={target!r}", flush=True)
        frames = load_video_frames(video)
        masks  = load_mask_frames(mask)

        edited = []
        import time; t0 = time.time()
        for fi, (fr, mk) in enumerate(zip(frames, masks)):
            ef = edit_frame(pipe, config.get("model", {}), fr, mk, target, font, gen)
            edited.append(ef)
            if fi == 0:
                print(f"  first frame done in {time.time()-t0:.1f}s", flush=True)
        write_mp4(edited, out_path, fps=24)
        print(f"  saved {out_path} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
