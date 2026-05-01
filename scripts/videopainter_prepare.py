"""
Prepare VideoPainter inputs for our 160-clip test set:
  1. Convert each text_mask mp4 -> all_masks.npz (binary, mask_id=1)
  2. Build a CSV with columns (video_path, mask_npz_path, fps, mask_id, start_frame, end_frame, caption)
  3. Use prompt template: 'Text that reads "<target_text>".'
"""
import csv, json
from pathlib import Path
import numpy as np
import cv2

REPO = Path("/home/xinghao/DiffSynth-Studio-TextVACE").resolve()
INF = REPO / "data" / "inference_new" / "inference_160"
META = INF / "metadata.csv"
WORK = REPO / "outputs" / "videopainter_work"
NPZ_DIR = WORK / "masks_npz"
CSV_OUT = WORK / "videopainter_inputs.csv"


def mp4_to_npz(mp4_path: Path, npz_path: Path) -> int:
    cap = cv2.VideoCapture(str(mp4_path))
    masks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = (gray > 127).astype(np.uint8)  # 1 where text region
        masks.append(binary)
    cap.release()
    arr = np.stack(masks, axis=0)  # (T, H, W) uint8
    np.savez(str(npz_path), arr_0=arr)
    return arr.shape[0]


def main():
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(META)))
    rows.sort(key=lambda r: r["id"])

    out_rows = []
    for r in rows:
        rid = r["id"]
        video_path = INF / r["vace_video"]      # original_videos/{id}.mp4
        mask_path = INF / r["vace_video_mask"]  # text_masks/{id}_<ts>_<hash>.mp4
        if not video_path.exists() or not mask_path.exists():
            print(f"  SKIP {rid}: missing file")
            continue
        npz_path = NPZ_DIR / f"{rid}.npz"
        if not npz_path.exists():
            n = mp4_to_npz(mask_path, npz_path)
        else:
            n = np.load(str(npz_path))["arr_0"].shape[0]
        prompt = f'Text that reads "{r["target_text"]}".'
        out_rows.append({
            "video_path":    str(video_path),
            "mask_npz_path": str(npz_path),
            "fps":           24,
            "mask_id":       1,
            "start_frame":   0,
            "end_frame":     n,    # end_frame is exclusive; use total frames
            "caption":       prompt,
            "id":            rid,
            "target_text":   r["target_text"],
            "first_frame_path": str(INF / "inference_160_textctrl_firstframes" / f"{rid}.png"),
        })

    fieldnames = list(out_rows[0].keys())
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(out_rows)
    print(f"wrote {CSV_OUT} ({len(out_rows)} rows)")
    print(f"wrote {len(list(NPZ_DIR.glob('*.npz')))} npz mask files in {NPZ_DIR}")


if __name__ == "__main__":
    main()
