"""
OCR Extraction for Benchmark Evaluation (runs in paddleocr conda env)
=====================================================================
Extracts per-frame OCR results from edited videos using PP-OCRv5.

For each frame: read the corresponding mask frame → compute bbox → crop → OCR.
Results saved to JSON for downstream metric computation.

Usage:
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  conda run -n paddleocr python scripts/benchmark/ocr_extract.py \
      --video_dir outputs/some_method/ \
      --mask_dir data/raw/text_masks \
      --records data/processed/parsed_records.json \
      --output ocr_results.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


def get_mask_bbox(mask_bin, pad_ratio=0.1):
    """Get bounding box of mask region with padding."""
    ys, xs = np.where(mask_bin > 0)
    if len(ys) == 0:
        return None
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    h, w = mask_bin.shape[:2]
    pad_h = int((y2 - y1) * pad_ratio)
    pad_w = int((x2 - x1) * pad_ratio)
    y1 = max(0, y1 - pad_h)
    y2 = min(h, y2 + pad_h)
    x1 = max(0, x1 - pad_w)
    x2 = min(w, x2 + pad_w)

    return x1, y1, x2, y2


def run_ocr_on_video(ocr, video_path, mask_path):
    """Run OCR on each frame, cropping to that frame's mask bbox.

    Uses per-frame mask to handle text that moves across frames.
    Returns list of OCR text strings, one per frame.
    """
    cap_v = cv2.VideoCapture(video_path)
    cap_m = cv2.VideoCapture(mask_path)
    total_frames = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))
    total_mask_frames = int(cap_m.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read edited video resolution
    v_w = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_h = int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    m_w = int(cap_m.get(cv2.CAP_PROP_FRAME_WIDTH))
    m_h = int(cap_m.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ocr_results = []
    for fi in range(total_frames):
        ret_v, frame = cap_v.read()
        if not ret_v:
            ocr_results.append("")
            continue

        # Read corresponding mask frame (loop if mask has fewer frames)
        if fi < total_mask_frames:
            ret_m, mask_frame = cap_m.read()
        else:
            # Reuse last mask frame
            ret_m = True

        if not ret_m or mask_frame is None:
            ocr_results.append("")
            continue

        # Resize mask to match video resolution if needed
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        if (m_h, m_w) != (v_h, v_w):
            mask_gray = cv2.resize(mask_gray, (v_w, v_h), interpolation=cv2.INTER_NEAREST)
        _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

        bbox = get_mask_bbox(mask_bin)
        if bbox is None:
            ocr_results.append("")
            continue

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            ocr_results.append("")
            continue

        # Run OCR
        texts = []
        try:
            for result in ocr.predict(crop):
                if hasattr(result, 'rec_texts') and result.rec_texts:
                    texts.extend(result.rec_texts)
        except Exception as e:
            print(f"    OCR error on frame {fi}: {e}")

        ocr_results.append(" ".join(texts))

    cap_v.release()
    cap_m.release()
    return ocr_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True,
                        help="Directory containing edited videos (*.mp4)")
    parser.add_argument("--mask_dir", default="data/raw/text_masks",
                        help="Directory containing mask videos")
    parser.add_argument("--records", default="data/processed/parsed_records.json",
                        help="Parsed records JSON")
    parser.add_argument("--output", required=True,
                        help="Output JSON file for OCR results")
    parser.add_argument("--lang", default="ch", help="OCR language")
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode for PaddlePaddle
    from paddleocr import PaddleOCR
    print("Initializing PP-OCRv5 (CPU mode)...")
    ocr = PaddleOCR(use_textline_orientation=True, lang=args.lang, device="cpu")

    with open(args.records) as f:
        records = json.load(f)
    records_map = {r["id"]: r for r in records}

    # Find videos in video_dir
    video_files = sorted([f for f in os.listdir(args.video_dir) if f.endswith(".mp4")])
    print(f"Found {len(video_files)} videos in {args.video_dir}")

    all_results = {}
    for i, vf in enumerate(video_files):
        vid_id = vf.replace("_generated.mp4", "").replace(".mp4", "")

        # Find mask
        mask_path = os.path.join(args.mask_dir, vid_id + ".mp4")
        if not os.path.exists(mask_path):
            print(f"  [{i+1}] {vid_id}: SKIP (no mask)")
            continue

        video_path = os.path.join(args.video_dir, vf)
        ocr_texts = run_ocr_on_video(ocr, video_path, mask_path)

        rec = records_map.get(vid_id, {})
        target_text = rec.get("target_text", "") or rec.get("replacement_text", "")

        all_results[vid_id] = {
            "ocr_per_frame": ocr_texts,
            "target_text": target_text,
            "num_frames": len(ocr_texts),
        }

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(video_files)}] {vid_id}: "
                  f"{len(ocr_texts)} frames, target=\"{target_text}\"")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved OCR results to {args.output}")
    print(f"Processed {len(all_results)} videos")


if __name__ == "__main__":
    main()
