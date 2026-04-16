"""
OCR Extraction (runs in paddleocr conda env, CPU mode)
=======================================================
For each edited video: reads per-frame mask, crops the mask bbox region
in the corresponding frame of the edited video, runs PP-OCRv5 on the crop,
and saves recognized text per frame to a JSON file.

This JSON is then consumed by Axis 1 metrics (SeqAcc, CharAcc, TTS).

Usage:
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  conda run -n paddleocr python scripts/benchmark/ocr_extract.py \
      --video_dir outputs/<method>/ \
      --mask_dir data/raw/text_masks/ \
      --records data/processed/parsed_records.json \
      --output outputs/<method>/ocr_results.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_mask_bbox


def run_ocr_on_video(ocr, video_path, mask_path):
    """Run OCR per-frame, cropping to that frame's mask bbox."""
    cap_v = cv2.VideoCapture(video_path)
    cap_m = cv2.VideoCapture(mask_path)
    total_frames = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))
    total_mask_frames = int(cap_m.get(cv2.CAP_PROP_FRAME_COUNT))
    v_w = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_h = int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    m_w = int(cap_m.get(cv2.CAP_PROP_FRAME_WIDTH))
    m_h = int(cap_m.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ocr_results = []
    last_mask_frame = None

    for fi in range(total_frames):
        ret_v, frame = cap_v.read()
        if not ret_v:
            ocr_results.append("")
            continue

        # Read per-frame mask; reuse last if exhausted
        if fi < total_mask_frames:
            ret_m, mask_frame = cap_m.read()
            if ret_m:
                last_mask_frame = mask_frame
            else:
                mask_frame = last_mask_frame
        else:
            mask_frame = last_mask_frame

        if mask_frame is None:
            ocr_results.append("")
            continue

        # Binarize and resize mask to match video resolution
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
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--mask_dir", default="data/raw/text_masks")
    parser.add_argument("--records", default="data/processed/parsed_records.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lang", default="ch")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU for PaddlePaddle
    from paddleocr import PaddleOCR
    print("Initializing PP-OCRv5 (CPU mode)...")
    ocr = PaddleOCR(use_textline_orientation=True, lang=args.lang, device="cpu")

    with open(args.records) as f:
        records = json.load(f)
    records_map = {r["id"]: r for r in records}

    video_files = sorted([f for f in os.listdir(args.video_dir) if f.endswith(".mp4")])
    print(f"Found {len(video_files)} videos in {args.video_dir}")

    all_results = {}
    for i, vf in enumerate(video_files):
        vid_id = vf.replace("_generated.mp4", "").replace(".mp4", "")
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

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(video_files)}] {vid_id}: "
                  f"{len(ocr_texts)}f, target=\"{target_text}\"")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved OCR to {args.output} ({len(all_results)} videos)")


if __name__ == "__main__":
    main()
