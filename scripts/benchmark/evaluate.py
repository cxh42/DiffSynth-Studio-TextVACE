"""
ReWording Benchmark Evaluation
===============================
Computes all 9 metrics across 3 axes for a video text editing method.

Axis 1 (Text Correctness): SeqAcc, CharAcc, TTS — requires pre-extracted OCR results
Axis 2 (Visual Quality): Flickering↑, MUSIQ↑, FVD↓
Axis 3 (Context Fidelity): PSNR↑, SSIM↑, LPIPS↓

Usage:
  # Step 1: Extract OCR results (in paddleocr env)
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  conda run -n paddleocr python scripts/benchmark/ocr_extract.py \
      --video_dir outputs/method_name/ --output outputs/method_name/ocr_results.json

  # Step 2: Run evaluation (in DiffSynth-Studio env)
  python scripts/benchmark/evaluate.py \
      --video_dir outputs/method_name/ \
      --ocr_results outputs/method_name/ocr_results.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Axis 1: Text Correctness (from pre-extracted OCR results)
# ---------------------------------------------------------------------------

def levenshtein_distance(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[-1]


def compute_text_correctness(ocr_results):
    """Compute SeqAcc, CharAcc, TTS from OCR results.

    Normalization: both OCR output and target are stripped and uppercased.
    """
    all_seq_acc = []
    all_char_acc = []
    all_tts = []
    per_sample = {}

    for vid_id, data in ocr_results.items():
        ocr_frames = data["ocr_per_frame"]
        target = data["target_text"].strip().upper()
        n = len(ocr_frames)
        if n == 0:
            continue

        # Normalize OCR outputs: strip + uppercase
        ocr_norm = [t.strip().upper() for t in ocr_frames]

        # SeqAcc: exact match per frame
        seq_matches = [1.0 if ocr_norm[i] == target else 0.0 for i in range(n)]
        seq_acc = np.mean(seq_matches)

        # CharAcc: 1 - edit_distance / max(len_pred, len_target)
        char_accs = []
        for i in range(n):
            pred = ocr_norm[i]
            if len(pred) == 0 and len(target) == 0:
                char_accs.append(1.0)
            elif len(pred) == 0 or len(target) == 0:
                char_accs.append(0.0)
            else:
                ed = levenshtein_distance(pred, target)
                char_accs.append(1.0 - ed / max(len(pred), len(target)))
        char_acc = np.mean(char_accs)

        # TTS: adjacent frame OCR consistency
        if n > 1:
            tts_matches = [1.0 if ocr_norm[i] == ocr_norm[i + 1] else 0.0
                           for i in range(n - 1)]
            tts = np.mean(tts_matches)
        else:
            tts = 1.0

        per_sample[vid_id] = {"SeqAcc": seq_acc, "CharAcc": char_acc, "TTS": tts}
        all_seq_acc.append(seq_acc)
        all_char_acc.append(char_acc)
        all_tts.append(tts)

    return {
        "aggregate": {
            "SeqAcc": float(np.mean(all_seq_acc)) if all_seq_acc else 0.0,
            "CharAcc": float(np.mean(all_char_acc)) if all_char_acc else 0.0,
            "TTS": float(np.mean(all_tts)) if all_tts else 0.0,
        },
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Video / mask loading utilities
# ---------------------------------------------------------------------------

def load_video_frames(video_path):
    """Load all frames as RGB numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_mask_frames(mask_path, target_h=None, target_w=None):
    """Load ALL mask frames as binary arrays, optionally resizing.

    Returns list of binary masks (H, W), one per frame.
    """
    cap = cv2.VideoCapture(mask_path)
    masks = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if target_h is not None and target_w is not None:
            if gray.shape[0] != target_h or gray.shape[1] != target_w:
                gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        _, mask_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        masks.append(mask_bin)
    cap.release()
    return masks


# ---------------------------------------------------------------------------
# Axis 2: Visual Quality (full frame)
# ---------------------------------------------------------------------------

def compute_flickering(edited_frames):
    """Compute Flickering score (VBench style).

    Uses cv2.absdiff between adjacent frames, normalized to 0-1:
    score = (255 - mean_MAE) / 255. Higher = more stable.
    """
    n = len(edited_frames)
    if n < 2:
        return 1.0

    maes = []
    for i in range(n - 1):
        diff = cv2.absdiff(edited_frames[i], edited_frames[i + 1])
        maes.append(np.mean(diff))

    mean_mae = float(np.mean(maes))
    return (255.0 - mean_mae) / 255.0


def compute_musiq(edited_frames, musiq_model):
    """Compute MUSIQ score on full frames (VBench style).

    Resizes to max 512px dimension before scoring.
    """
    scores = []
    device = next(musiq_model.parameters()).device

    for frame in edited_frames:
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        if max_dim > 512:
            scale = 512.0 / max_dim
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        frame_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        frame_t = frame_t.to(device)
        with torch.no_grad():
            score = musiq_model(frame_t)
        scores.append(float(score.item()))

    return float(np.mean(scores)) if scores else 0.0


def compute_fvd(edited_videos_frames, gt_videos_frames):
    """Compute FVD between edited and GT video sets using R3D-18 features."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from scripts.benchmark.fvd_utils import compute_fvd_from_videos
        return compute_fvd_from_videos(edited_videos_frames, gt_videos_frames)
    except Exception as e:
        print(f"  WARNING: FVD computation failed ({e}). Skipping.")
        return -1.0


# ---------------------------------------------------------------------------
# Axis 3: Context Fidelity (non-text region, per-frame mask)
# ---------------------------------------------------------------------------

def compute_psnr_masked(orig_frame, edited_frame, mask_inv):
    """Compute PSNR on non-text (mask-inverted) region."""
    pixels_orig = orig_frame[mask_inv].astype(np.float64)
    pixels_edit = edited_frame[mask_inv].astype(np.float64)
    if pixels_orig.size == 0:
        return 100.0
    mse = np.mean((pixels_orig - pixels_edit) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def compute_ssim_masked(orig_frame, edited_frame, mask_inv):
    """Compute SSIM on non-text region.

    Sets text region to original pixel values in BOTH images so the
    text area contributes identically and doesn't affect the comparison.
    """
    from skimage.metrics import structural_similarity
    o = orig_frame.copy()
    e = edited_frame.copy()
    mask_text = ~mask_inv
    # Set text region in edited frame to original values → cancels out
    e[mask_text] = o[mask_text]
    return float(structural_similarity(o, e, channel_axis=2, data_range=255))


def compute_lpips_masked(orig_frame, edited_frame, mask_inv, lpips_model):
    """Compute LPIPS with text region set to original values.

    By setting text pixels to the same (original) values in both images,
    the text area contributes zero distance and only the background
    differences are captured.
    """
    o = orig_frame.copy()
    e = edited_frame.copy()
    mask_text = ~mask_inv
    e[mask_text] = o[mask_text]

    device = next(lpips_model.parameters()).device
    o_t = torch.from_numpy(o).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0
    e_t = torch.from_numpy(e).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    with torch.no_grad():
        d = lpips_model(o_t, e_t)
    return float(d.item())


def compute_context_fidelity_perframe(orig_frames, edited_frames, mask_frames, lpips_model):
    """Compute PSNR, SSIM, LPIPS on non-text region with per-frame masks."""
    psnrs, ssims, lpipss = [], [], []
    n = len(edited_frames)

    for i in range(n):
        # Use per-frame mask; fall back to last mask if fewer mask frames
        mask_idx = min(i, len(mask_frames) - 1)
        mask_inv = mask_frames[mask_idx] == 0  # non-text region

        psnrs.append(compute_psnr_masked(orig_frames[i], edited_frames[i], mask_inv))
        ssims.append(compute_ssim_masked(orig_frames[i], edited_frames[i], mask_inv))
        lpipss.append(compute_lpips_masked(orig_frames[i], edited_frames[i], mask_inv, lpips_model))

    return {
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
        "LPIPS": float(np.mean(lpipss)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True,
                        help="Directory with edited videos (*.mp4)")
    parser.add_argument("--ocr_results", default=None,
                        help="OCR results JSON from ocr_extract.py")
    parser.add_argument("--orig_dir", default="data/raw/original_videos",
                        help="Original videos directory")
    parser.add_argument("--gt_dir", default="data/raw/edited_videos",
                        help="Ground truth edited videos directory")
    parser.add_argument("--mask_dir", default="data/raw/text_masks",
                        help="Mask videos directory")
    parser.add_argument("--records", default="data/processed/parsed_records.json")
    parser.add_argument("--output", default=None,
                        help="Output JSON (default: video_dir/eval_results.json)")
    parser.add_argument("--skip_fvd", action="store_true", help="Skip FVD")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.video_dir, "eval_results.json")

    with open(args.records) as f:
        records = json.load(f)
    records_map = {r["id"]: r for r in records}

    video_files = sorted([f for f in os.listdir(args.video_dir) if f.endswith(".mp4")])
    vid_ids = [f.replace("_generated.mp4", "").replace(".mp4", "") for f in video_files]
    print(f"Found {len(video_files)} videos to evaluate")

    results = {"per_sample": {}, "aggregate": {}}

    # ========================
    # Axis 1: Text Correctness
    # ========================
    if args.ocr_results and os.path.exists(args.ocr_results):
        print("\n=== Axis 1: Text Correctness ===")
        with open(args.ocr_results) as f:
            ocr_data = json.load(f)
        tc = compute_text_correctness(ocr_data)
        results["aggregate"].update(tc["aggregate"])
        for vid_id, scores in tc["per_sample"].items():
            results["per_sample"].setdefault(vid_id, {}).update(scores)
        print(f"  SeqAcc:  {tc['aggregate']['SeqAcc']:.4f}")
        print(f"  CharAcc: {tc['aggregate']['CharAcc']:.4f}")
        print(f"  TTS:     {tc['aggregate']['TTS']:.4f}")
    else:
        print("\n=== Axis 1: SKIPPED (no OCR results) ===")

    # ========================
    # Axis 2 & 3
    # ========================
    print("\n=== Loading models ===")
    import pyiqa
    musiq_model = pyiqa.create_metric("musiq", device=args.device)
    print("  MUSIQ loaded")

    import lpips
    lpips_model = lpips.LPIPS(net="alex").to(args.device)
    print("  LPIPS loaded")

    all_flickering, all_musiq = [], []
    all_psnr, all_ssim, all_lpips_scores = [], [], []
    all_edited_for_fvd, all_gt_for_fvd = [], []

    print(f"\n=== Processing {len(video_files)} videos ===")
    for i, (vf, vid_id) in enumerate(zip(video_files, vid_ids)):
        video_path = os.path.join(args.video_dir, vf)
        orig_path = os.path.join(args.orig_dir, vid_id + ".mp4")
        gt_path = os.path.join(args.gt_dir, vid_id + ".mp4")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(args.gt_dir, vid_id + "_overlay.mp4")
        mask_path = os.path.join(args.mask_dir, vid_id + ".mp4")

        if not os.path.exists(orig_path) or not os.path.exists(mask_path):
            continue

        # Load frames
        edited_frames = load_video_frames(video_path)
        orig_frames = load_video_frames(orig_path)

        if len(edited_frames) == 0 or len(orig_frames) == 0:
            continue

        # Align frame counts
        n = min(len(edited_frames), len(orig_frames))
        edited_frames = edited_frames[:n]
        orig_frames = orig_frames[:n]

        # Resize orig to match edited if needed
        eh, ew = edited_frames[0].shape[:2]
        oh, ow = orig_frames[0].shape[:2]
        if (eh, ew) != (oh, ow):
            orig_frames = [cv2.resize(f, (ew, eh)) for f in orig_frames]

        # Load per-frame masks, resized to edited resolution
        mask_frames = load_mask_frames(mask_path, target_h=eh, target_w=ew)

        # --- Axis 2: Visual Quality (full frame) ---
        flicker = compute_flickering(edited_frames)
        musiq_score = compute_musiq(edited_frames, musiq_model)

        all_flickering.append(flicker)
        all_musiq.append(musiq_score)

        # Collect for FVD
        if os.path.exists(gt_path):
            gt_frames = load_video_frames(gt_path)
            if gt_frames:
                gh, gw = gt_frames[0].shape[:2]
                if (eh, ew) != (gh, gw):
                    gt_frames = [cv2.resize(f, (ew, eh)) for f in gt_frames]
                all_edited_for_fvd.append(edited_frames)
                all_gt_for_fvd.append(gt_frames[:n])

        # --- Axis 3: Context Fidelity (per-frame mask) ---
        ctx = compute_context_fidelity_perframe(
            orig_frames, edited_frames, mask_frames, lpips_model
        )

        all_psnr.append(ctx["PSNR"])
        all_ssim.append(ctx["SSIM"])
        all_lpips_scores.append(ctx["LPIPS"])

        results["per_sample"].setdefault(vid_id, {}).update({
            "Flickering": flicker,
            "MUSIQ": musiq_score,
            "PSNR": ctx["PSNR"],
            "SSIM": ctx["SSIM"],
            "LPIPS": ctx["LPIPS"],
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(video_files)}] {vid_id}: "
                  f"Flicker={flicker:.4f} MUSIQ={musiq_score:.1f} "
                  f"PSNR={ctx['PSNR']:.1f} SSIM={ctx['SSIM']:.3f} LPIPS={ctx['LPIPS']:.4f}")

    # Aggregates
    results["aggregate"]["Flickering"] = float(np.mean(all_flickering)) if all_flickering else 0.0
    results["aggregate"]["MUSIQ"] = float(np.mean(all_musiq)) if all_musiq else 0.0

    if not args.skip_fvd and all_edited_for_fvd:
        print("\n=== Computing FVD ===")
        fvd = compute_fvd(all_edited_for_fvd, all_gt_for_fvd)
        results["aggregate"]["FVD"] = fvd
        print(f"  FVD: {fvd:.2f}")
    else:
        results["aggregate"]["FVD"] = -1.0
        print("\n=== FVD: SKIPPED ===")

    results["aggregate"]["PSNR"] = float(np.mean(all_psnr)) if all_psnr else 0.0
    results["aggregate"]["SSIM"] = float(np.mean(all_ssim)) if all_ssim else 0.0
    results["aggregate"]["LPIPS"] = float(np.mean(all_lpips_scores)) if all_lpips_scores else 0.0

    # Print summary
    print(f"\n{'='*60}")
    print(f"ReWording Benchmark Results ({len(video_files)} videos)")
    print(f"{'='*60}")
    print(f"\nAxis 1 - Text Correctness:")
    print(f"  SeqAcc:     {results['aggregate'].get('SeqAcc', 'N/A')}")
    print(f"  CharAcc:    {results['aggregate'].get('CharAcc', 'N/A')}")
    print(f"  TTS:        {results['aggregate'].get('TTS', 'N/A')}")
    print(f"\nAxis 2 - Visual Quality:")
    print(f"  Flickering: {results['aggregate']['Flickering']:.4f} (higher=more stable)")
    print(f"  MUSIQ:      {results['aggregate']['MUSIQ']:.2f} (higher=better)")
    print(f"  FVD:        {results['aggregate']['FVD']:.2f} (lower=better)")
    print(f"\nAxis 3 - Context Fidelity:")
    print(f"  PSNR:       {results['aggregate']['PSNR']:.2f} dB (higher=better)")
    print(f"  SSIM:       {results['aggregate']['SSIM']:.4f} (higher=better)")
    print(f"  LPIPS:      {results['aggregate']['LPIPS']:.4f} (lower=better)")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
