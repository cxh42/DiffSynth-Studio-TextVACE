"""
Generate per-GPU sharded configs for AnyV2V on the 160-clip test set.
Writes:
  baselines/AnyV2V/i2vgen-xl/configs/group_ddim_inversion/shard{0..6}.{json,yaml}
  baselines/AnyV2V/i2vgen-xl/configs/group_pnp_edit/shard{0..6}.{json,yaml}
"""
import csv, json
from pathlib import Path

REPO_ROOT = Path("/home/xinghao/DiffSynth-Studio-TextVACE").resolve()
ANYV2V_DIR = REPO_ROOT / "baselines" / "AnyV2V" / "i2vgen-xl"
META = REPO_ROOT / "data" / "inference_new" / "inference_160" / "metadata.csv"
VIDEO_DIR = REPO_ROOT / "data" / "inference_new" / "inference_160" / "original_videos"
FF_DIR = REPO_ROOT / "data" / "inference_new" / "inference_160" / "inference_160_textctrl_firstframes"
WORK_DIR = REPO_ROOT / "outputs" / "anyv2v_work"
IMAGE_SIZE = [512, 512]    # (W, H); 720p OOMs PnP edit on H100 80GB
GPUS = [1, 2, 3, 4, 5, 6, 7]


def main():
    rows = list(csv.DictReader(open(META)))
    rows.sort(key=lambda r: r["id"])
    print(f"loaded {len(rows)} samples")

    inv_dir = ANYV2V_DIR / "configs" / "group_ddim_inversion"
    pnp_dir = ANYV2V_DIR / "configs" / "group_pnp_edit"
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    for shard_idx, gpu in enumerate(GPUS):
        shard_rows = [r for i, r in enumerate(rows) if i % len(GPUS) == shard_idx]
        # ---- DDIM inversion shard ----
        inv_entries = []
        for r in shard_rows:
            inv_entries.append({
                "active": True,
                "force_recompute_latents": False,
                "video_name": r["id"],
                "image_size": list(IMAGE_SIZE),
                "recon_config": {"enable_recon": False},
            })
        inv_template = f"""# auto-generated shard {shard_idx} (GPU {gpu})
seed: 8888
device: "cuda:0"   # CUDA_VISIBLE_DEVICES isolates to GPU {gpu}
debug: False

data_dir: "{REPO_ROOT}"
model_name: "i2vgen-xl"
exp_name: "${{video_name}}"
output_dir: "{WORK_DIR}/inversions/${{video_name}}"

image_size: {IMAGE_SIZE}
video_dir: "{VIDEO_DIR}"
video_name: "ReplaceMe"
video_path: "ReplaceMe"
video_frames_path: "ReplaceMe"

n_frames: 120

inverse_config:
    image_size: ${{image_size}}
    n_frames: ${{n_frames}}
    cfg: 1.0
    target_fps: 24
    prompt: ""
    negative_prompt: ""
    n_steps: 500
    output_dir: "${{output_dir}}/ddim_latents"
    inverse_static_video: False
    null_image_inversion: False

recon_config:
    enable_recon: False
    image_size: ${{image_size}}
    n_frames: ${{n_frames}}
    cfg: 9.0
    target_fps: 24
    prompt: ""
    negative_prompt: "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
    n_steps: 50
    ddim_init_latents_t_idx: 3
    ddim_latents_path: "${{inverse_config.output_dir}}"
"""
        (inv_dir / f"shard{shard_idx}.json").write_text(json.dumps(inv_entries, indent=2))
        (inv_dir / f"shard{shard_idx}.yaml").write_text(inv_template)

        # ---- PnP edit shard ----
        pnp_entries = []
        for r in shard_rows:
            ff_path = FF_DIR / f"{r['id']}.png"
            if not ff_path.exists():
                print(f"  WARN: missing first frame: {ff_path}")
                continue
            edit_prompt = f"Change {r['source_text']} to {r['target_text']}; preserve everything else."
            pnp_entries.append({
                "active": True,
                "task_name": "Prompt-Based-Editing",
                "video_name": r["id"],
                "edited_first_frame_path": str(ff_path),
                "editing_prompt": edit_prompt,
                "edited_video_name": r["id"],
                "image_size": list(IMAGE_SIZE),
            })
        pnp_template = f"""# auto-generated shard {shard_idx} (GPU {gpu})
seed: 8888
device: "cuda:0"   # CUDA_VISIBLE_DEVICES isolates to GPU {gpu}
debug: False

data_dir: "{REPO_ROOT}"
model_name: "i2vgen-xl"
task_name: "Prompt-Based-Editing"
edited_video_name: "ReplaceMe"
output_dir: "{WORK_DIR}/results/${{video_name}}/${{edited_video_name}}/"

image_size: {IMAGE_SIZE}
video_dir: "{VIDEO_DIR}"
video_name: "ReplaceMe"
video_path: "ReplaceMe"
video_frames_path: "ReplaceMe"
edited_first_frame_path: "ReplaceMe"
ddim_latents_path: "{WORK_DIR}/inversions/${{video_name}}/ddim_latents"

n_frames: 120
cfg: 9.0
target_fps: 24
editing_prompt: "ReplaceMe"
editing_negative_prompt: "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
n_steps: 50
ddim_init_latents_t_idx: 1
ddim_inv_prompt: ""
random_ratio: 0.0

pnp_f_t: 0.2
pnp_spatial_attn_t: 0.2
pnp_temp_attn_t: 0.5
"""
        (pnp_dir / f"shard{shard_idx}.json").write_text(json.dumps(pnp_entries, indent=2))
        (pnp_dir / f"shard{shard_idx}.yaml").write_text(pnp_template)

        print(f"shard {shard_idx} (GPU {gpu}): {len(shard_rows)} samples -> {len(pnp_entries)} valid")

    print(f"\nconfig dirs:\n  {inv_dir}\n  {pnp_dir}")
    print(f"work dir: {WORK_DIR}")


if __name__ == "__main__":
    main()
