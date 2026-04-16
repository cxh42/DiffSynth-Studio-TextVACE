"""
ReWording Benchmark Orchestrator
==================================
Runs all 9 metrics across 3 axes and writes a consolidated result JSON.

Axis 1: SeqAcc, CharAcc, TTS     (from pre-extracted OCR results)
Axis 2: Flickering, MUSIQ, FVD   (visual quality, FVD needs reference videos)
Axis 3: PSNR, SSIM, LPIPS        (non-text region vs original video)

Usage:
  # Step 1: Extract OCR (paddleocr env)
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  conda run -n paddleocr python scripts/benchmark/ocr_extract.py \
      --video_dir outputs/<method>/ \
      --mask_dir <mask_dir>/ \
      --records <records.json> \
      --output outputs/<method>/ocr_results.json

  # Step 2: Run all metrics (DiffSynth-Studio env)
  python scripts/benchmark/evaluate.py \
      --video_dir outputs/<method>/ \
      --ocr_results outputs/<method>/ocr_results.json \
      --orig_dir <orig_dir>/ \
      --mask_dir <mask_dir>/ \
      --ref_dir data/raw/edited_videos/
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_video_frames, load_mask_frames

import metric_seq_acc
import metric_char_acc
import metric_tts
import metric_flickering
import metric_musiq
import metric_fvd
import metric_psnr
import metric_ssim
import metric_lpips


def run_axis1(ocr_results_path):
    """Axis 1: Text Correctness."""
    if not ocr_results_path or not os.path.exists(ocr_results_path):
        return None, None

    with open(ocr_results_path) as f:
        ocr_data = json.load(f)

    per_sample = {}
    seq_all, char_all, tts_all = [], [], []

    for vid_id, data in ocr_data.items():
        ocr = data["ocr_per_frame"]
        target = data["target_text"]
        if not ocr:
            continue

        seq = metric_seq_acc.compute(ocr, target)
        char = metric_char_acc.compute(ocr, target)
        tts = metric_tts.compute(ocr)

        per_sample[vid_id] = {"SeqAcc": seq, "CharAcc": char, "TTS": tts}
        seq_all.append(seq)
        char_all.append(char)
        tts_all.append(tts)

    aggregate = {
        "SeqAcc": float(np.mean(seq_all)) if seq_all else 0.0,
        "CharAcc": float(np.mean(char_all)) if char_all else 0.0,
        "TTS": float(np.mean(tts_all)) if tts_all else 0.0,
    }
    return aggregate, per_sample


def find_gt_path(gt_dir, vid_id):
    """Find GT video allowing `_overlay` suffix variant."""
    p = os.path.join(gt_dir, vid_id + ".mp4")
    if os.path.exists(p):
        return p
    p = os.path.join(gt_dir, vid_id + "_overlay.mp4")
    if os.path.exists(p):
        return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True,
                        help="Directory with edited output videos (*.mp4)")
    parser.add_argument("--ocr_results", default=None,
                        help="OCR results JSON (from ocr_extract.py)")
    parser.add_argument("--orig_dir", default="data/raw/original_videos",
                        help="Original (pre-edit) videos directory")
    parser.add_argument("--mask_dir", default="data/raw/text_masks",
                        help="Text mask videos directory")
    parser.add_argument("--ref_dir", default="data/raw/edited_videos",
                        help="Reference high-quality edited videos (for FVD)")
    parser.add_argument("--output", default=None,
                        help="Output JSON (default: video_dir/eval_results.json)")
    parser.add_argument("--skip_fvd", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.video_dir, "eval_results.json")

    video_files = sorted([f for f in os.listdir(args.video_dir) if f.endswith(".mp4")])
    vid_ids = [f.replace("_generated.mp4", "").replace(".mp4", "") for f in video_files]
    print(f"Evaluating {len(video_files)} videos from {args.video_dir}")

    results = {"per_sample": {}, "aggregate": {}}

    # --- Axis 1 ---
    print("\n=== Axis 1: Text Correctness ===")
    agg1, per_sample1 = run_axis1(args.ocr_results)
    if agg1 is not None:
        results["aggregate"].update(agg1)
        for vid_id, scores in per_sample1.items():
            results["per_sample"].setdefault(vid_id, {}).update(scores)
        print(f"  SeqAcc:  {agg1['SeqAcc']:.4f}")
        print(f"  CharAcc: {agg1['CharAcc']:.4f}")
        print(f"  TTS:     {agg1['TTS']:.4f}")
    else:
        print("  SKIPPED (no OCR results)")

    # --- Axis 2 & 3 ---
    print("\n=== Axis 2 + 3: Loading videos ===")
    flicker_all, musiq_all = [], []
    psnr_all, ssim_all, lpips_all = [], [], []
    fvd_edited, fvd_ref = [], []

    for i, (vf, vid_id) in enumerate(zip(video_files, vid_ids)):
        video_path = os.path.join(args.video_dir, vf)
        orig_path = os.path.join(args.orig_dir, vid_id + ".mp4")
        mask_path = os.path.join(args.mask_dir, vid_id + ".mp4")

        if not os.path.exists(orig_path) or not os.path.exists(mask_path):
            print(f"  [{i+1}] {vid_id}: SKIP (missing orig/mask)")
            continue

        edited_frames = load_video_frames(video_path)
        orig_frames = load_video_frames(orig_path)
        if not edited_frames or not orig_frames:
            continue

        n = min(len(edited_frames), len(orig_frames))
        edited_frames = edited_frames[:n]
        orig_frames = orig_frames[:n]

        eh, ew = edited_frames[0].shape[:2]
        oh, ow = orig_frames[0].shape[:2]
        if (eh, ew) != (oh, ow):
            orig_frames = [cv2.resize(f, (ew, eh)) for f in orig_frames]

        mask_frames = load_mask_frames(mask_path, target_h=eh, target_w=ew)

        # --- Axis 2 (full frame) ---
        flicker = metric_flickering.compute(edited_frames)
        musiq = metric_musiq.compute(edited_frames, device=args.device)

        # --- Axis 3 (non-text region) ---
        psnr = metric_psnr.compute(orig_frames, edited_frames, mask_frames)
        ssim = metric_ssim.compute(orig_frames, edited_frames, mask_frames)
        lpips_s = metric_lpips.compute(orig_frames, edited_frames, mask_frames, device=args.device)

        flicker_all.append(flicker)
        musiq_all.append(musiq)
        psnr_all.append(psnr)
        ssim_all.append(ssim)
        lpips_all.append(lpips_s)

        # Collect edited video for FVD (reference distribution loaded separately below)
        fvd_edited.append(edited_frames)

        results["per_sample"].setdefault(vid_id, {}).update({
            "Flickering": flicker,
            "MUSIQ": musiq,
            "PSNR": psnr,
            "SSIM": ssim,
            "LPIPS": lpips_s,
        })

        print(f"  [{i+1}/{len(video_files)}] {vid_id}: "
              f"Flick={flicker:.4f} MUSIQ={musiq:.1f} "
              f"PSNR={psnr:.1f} SSIM={ssim:.3f} LPIPS={lpips_s:.4f}")

    # Aggregate
    results["aggregate"]["Flickering"] = float(np.mean(flicker_all)) if flicker_all else 0.0
    results["aggregate"]["MUSIQ"] = float(np.mean(musiq_all)) if musiq_all else 0.0
    results["aggregate"]["PSNR"] = float(np.mean(psnr_all)) if psnr_all else 0.0
    results["aggregate"]["SSIM"] = float(np.mean(ssim_all)) if ssim_all else 0.0
    results["aggregate"]["LPIPS"] = float(np.mean(lpips_all)) if lpips_all else 0.0

    # FVD: load all reference videos from ref_dir as the target distribution
    if not args.skip_fvd and len(fvd_edited) >= 2 and os.path.isdir(args.ref_dir):
        print(f"\n=== Computing FVD ===")
        ref_files = sorted([f for f in os.listdir(args.ref_dir) if f.endswith(".mp4")])
        print(f"  Loading {len(ref_files)} reference videos from {args.ref_dir}...")
        target_h, target_w = fvd_edited[0][0].shape[:2]
        fvd_ref = []
        for rf in ref_files:
            rframes = load_video_frames(os.path.join(args.ref_dir, rf))
            if not rframes:
                continue
            rh, rw = rframes[0].shape[:2]
            if (target_h, target_w) != (rh, rw):
                rframes = [cv2.resize(f, (target_w, target_h)) for f in rframes]
            fvd_ref.append(rframes)

        if len(fvd_ref) >= 2:
            print(f"  {len(fvd_edited)} edited vs {len(fvd_ref)} reference videos")
            try:
                fvd = metric_fvd.compute(fvd_edited, fvd_ref, device=args.device)
                results["aggregate"]["FVD"] = fvd
                print(f"  FVD: {fvd:.2f}")
            except Exception as e:
                print(f"  FVD failed: {e}")
                results["aggregate"]["FVD"] = -1.0
        else:
            results["aggregate"]["FVD"] = -1.0
            print(f"  FVD: SKIPPED (only {len(fvd_ref)} reference videos loaded)")
    else:
        results["aggregate"]["FVD"] = -1.0
        print(f"\n=== FVD: SKIPPED ===")

    # Print summary
    agg = results["aggregate"]
    print(f"\n{'='*60}")
    print(f"ReWording Benchmark Results ({len(video_files)} videos)")
    print(f"{'='*60}")
    print(f"Axis 1 - Text Correctness:")
    print(f"  SeqAcc:     {agg.get('SeqAcc', 'N/A')}")
    print(f"  CharAcc:    {agg.get('CharAcc', 'N/A')}")
    print(f"  TTS:        {agg.get('TTS', 'N/A')}")
    print(f"Axis 2 - Visual Quality:")
    print(f"  Flickering: {agg['Flickering']:.4f} (higher=more stable)")
    print(f"  MUSIQ:      {agg['MUSIQ']:.2f} (higher=better)")
    print(f"  FVD:        {agg['FVD']:.2f} (lower=better)")
    print(f"Axis 3 - Context Fidelity:")
    print(f"  PSNR:       {agg['PSNR']:.2f} dB (higher=better)")
    print(f"  SSIM:       {agg['SSIM']:.4f} (higher=better)")
    print(f"  LPIPS:      {agg['LPIPS']:.4f} (lower=better)")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
