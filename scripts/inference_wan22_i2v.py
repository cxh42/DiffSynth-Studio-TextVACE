"""
Wan2.2-I2V-A14B baseline (Family D variant): edited first frame + prompt -> video.
Sharded multi-GPU via --worker_rank / --num_workers.
"""

import os, sys, csv, glob, argparse, traceback
import torch
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video


MODEL_BASE = "models/Wan-AI/Wan2.2-I2V-A14B"
TOKENIZER_PATH = "models/Wan-AI/Wan2.2-I2V-A14B"  # tokenizer ships in the same repo

DATA_DIR_DEFAULT = "data/inference_new/inference_160"
META_DEFAULT = "data/inference_new/inference_160/metadata.csv"
FF_DIR_DEFAULT = "data/inference_new/inference_160/inference_160_textctrl_firstframes"
OUT_DEFAULT = "outputs/baseline_wan22_i2v_inference_160"

# Match GlyphVACE eval grid: 1280x720, 121 frames
HEIGHT = 720
WIDTH = 1280
NUM_FRAMES = 121
NUM_INFERENCE_STEPS = 50
CFG_SCALE = 5.0
SEED = 42
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def build_pipe(device, torch_dtype=torch.bfloat16):
    print(f"[wan2.2-i2v] loading pipeline on {device}", flush=True)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(path=sorted(glob.glob(os.path.join(MODEL_BASE, "high_noise_model/diffusion_pytorch_model*.safetensors")))),
            ModelConfig(path=sorted(glob.glob(os.path.join(MODEL_BASE, "low_noise_model/diffusion_pytorch_model*.safetensors")))),
            ModelConfig(path=os.path.join(MODEL_BASE, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=os.path.join(MODEL_BASE, "Wan2.1_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=os.path.join(TOKENIZER_PATH, "google/umt5-xxl")),
        redirect_common_files=False,
    )
    print(f"[wan2.2-i2v] pipe ready", flush=True)
    return pipe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default=DATA_DIR_DEFAULT)
    p.add_argument("--metadata",   default=META_DEFAULT)
    p.add_argument("--ff_dir",     default=FF_DIR_DEFAULT, help="dir containing edited first-frame PNGs ({id}.png)")
    p.add_argument("--output_dir", default=OUT_DEFAULT)
    p.add_argument("--worker_rank", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--height", type=int, default=HEIGHT)
    p.add_argument("--width",  type=int, default=WIDTH)
    p.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    p.add_argument("--prompt_field", default="instruction",
        help="CSV column to use as prompt; defaults to instruction (Family-B template)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu}"

    # Load metadata, shard
    rows = list(csv.DictReader(open(args.metadata)))
    rows = [r for i, r in enumerate(rows) if i % args.num_workers == args.worker_rank]
    print(f"[rank {args.worker_rank}/{args.num_workers}] {len(rows)} samples assigned", flush=True)

    pipe = build_pipe(device)

    target_size = (args.width, args.height)  # PIL resize order

    for idx, row in enumerate(rows):
        rid = row["id"]
        out_path = os.path.join(args.output_dir, f"{rid}.mp4")
        if os.path.exists(out_path):
            print(f"[rank {args.worker_rank}] [{idx+1}/{len(rows)}] SKIP {rid}", flush=True)
            continue

        ff_path = os.path.join(args.ff_dir, f"{rid}.png")
        if not os.path.exists(ff_path):
            print(f"[rank {args.worker_rank}] [{idx+1}/{len(rows)}] WARN missing first frame: {ff_path}", flush=True)
            continue

        prompt = row.get(args.prompt_field) or row["prompt"]
        print(f"[rank {args.worker_rank}] [{idx+1}/{len(rows)}] {rid}: {prompt}", flush=True)

        try:
            ff = Image.open(ff_path).convert("RGB").resize(target_size, Image.LANCZOS)
            video = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                input_image=ff,
                seed=SEED,
                tiled=True,
                switch_DiT_boundary=0.9,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                cfg_scale=CFG_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
            )
            save_video(video, out_path, fps=15, quality=5)
            print(f"[rank {args.worker_rank}]   Saved {out_path}", flush=True)
        except Exception as e:
            print(f"[rank {args.worker_rank}]   ERROR on {rid}: {e}", flush=True)
            traceback.print_exc()
            continue

    print(f"[rank {args.worker_rank}] done", flush=True)


if __name__ == "__main__":
    main()
