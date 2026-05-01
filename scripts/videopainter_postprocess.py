"""
Post-process VideoPainter outputs to evaluation-grid format:
  1. If clean {id}.mp4 exists → use directly; else crop the right pane (4-pane viz)
  2. Trim padded tail: 49 frames -> 40 frames (drop 9 repeated last)
  3. Frame-interpolate 8 fps -> 24 fps (40 -> 120 frames) via linear blend
  4. Spatial upscale 720x480 -> 1280x720 (Lanczos)
Output: outputs/baseline_videopainter_inference_160_postproc/{id}.mp4
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor

REPO = Path("/home/xinghao/DiffSynth-Studio-TextVACE")
SRC_DIR = REPO / "outputs" / "baseline_videopainter_inference_160"
OUT_DIR = REPO / "outputs" / "baseline_videopainter_inference_160_postproc"
TARGET_W, TARGET_H = 1280, 720
N_RAW = 49             # VideoPainter native output frame count (with padding)
N_KEEP = 40            # we drop the 9 repeated last frames
N_FINAL = 120          # 5s @ 24fps target
INTERP_FACTOR = 3      # 8fps -> 24fps


def read_mp4(path: Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return frames


def write_mp4(frames, path: Path, fps=24):
    if not frames: return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def linear_interpolate(frames, factor=3):
    """Insert (factor-1) blended frames between each adjacent pair."""
    if factor <= 1: return frames
    out = []
    for i in range(len(frames) - 1):
        a, b = frames[i].astype(np.float32), frames[i+1].astype(np.float32)
        out.append(frames[i])
        for k in range(1, factor):
            t = k / factor
            blend = (a * (1 - t) + b * t).astype(np.uint8)
            out.append(blend)
    out.append(frames[-1])
    return out


def process_one(rid: str):
    clean = SRC_DIR / f"{rid}.mp4"
    viz = SRC_DIR / f"{rid}_fps8.mp4"
    out_path = OUT_DIR / f"{rid}.mp4"
    if out_path.exists():
        return rid, "skip"

    if clean.exists():
        frames = read_mp4(clean)
    elif viz.exists():
        # 4-pane horizontally concatenated; take right-most (edit) pane.
        f0 = read_mp4(viz)
        if not f0: return rid, "empty viz"
        h, w = f0[0].shape[:2]
        pane_w = w // 4  # 4 panes
        frames = [f[:, 3*pane_w:4*pane_w] for f in f0]
    else:
        return rid, "missing"

    # Trim padded tail (49 -> 40)
    if len(frames) >= N_RAW:
        frames = frames[:N_KEEP]

    # Frame-interpolate 8fps -> 24fps (40 -> 118 frames)
    frames = linear_interpolate(frames, factor=INTERP_FACTOR)
    # Pad/truncate to exact 120 frames
    if len(frames) < N_FINAL:
        frames = frames + [frames[-1]] * (N_FINAL - len(frames))
    frames = frames[:N_FINAL]

    # Spatial upscale to 1280x720
    frames = [cv2.resize(f, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4) for f in frames]

    write_mp4(frames, out_path, fps=24)
    return rid, "ok"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Get list of IDs from clean or viz mp4s
    ids = set()
    for p in SRC_DIR.glob("*.mp4"):
        rid = p.stem.replace("_fps8", "")
        ids.add(rid)
    ids = sorted(ids)
    print(f"processing {len(ids)} samples (parallel)")
    with ProcessPoolExecutor(max_workers=8) as ex:
        for rid, status in ex.map(process_one, ids):
            print(f"  {rid}: {status}")
    print(f"DONE -> {OUT_DIR}")


if __name__ == "__main__":
    main()
