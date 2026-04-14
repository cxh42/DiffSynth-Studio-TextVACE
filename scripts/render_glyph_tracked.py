"""
High-Quality Glyph Video Rendering via EasyOCR + CoTracker2
=============================================================
Step 1: EasyOCR detects text precisely in the first frame (tight quadrilateral)
Step 2: CoTracker2 tracks the 4 corner points across all video frames
Step 3: Render target text with perspective transform at tracked positions

This produces temporally smooth, perspective-accurate glyph videos.

Usage:
  conda run -n DiffSynth-Studio python scripts/render_glyph_tracked.py
  conda run -n DiffSynth-Studio python scripts/render_glyph_tracked.py --sample_id 0000007_00000
"""

import argparse
import json
import os
import sys

import cv2
import easyocr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.prepare_textvace_data import _detect_script, _SCRIPT_FONTS


# ---------------------------------------------------------------------------
# EasyOCR: precise first-frame text detection
# ---------------------------------------------------------------------------

def init_ocr(langs=("en", "ch_sim")):
    """Initialize EasyOCR reader."""
    return easyocr.Reader(list(langs), gpu=True, verbose=False)


def detect_text_first_frame(reader, frame_rgb, mask_bin, source_text=None):
    """Detect text in a frame, find the one inside the mask.

    Returns 4 corner points (tight quadrilateral) or None.
    """
    results = reader.readtext(frame_rgb)
    if not results:
        return None

    best = None
    best_score = -1

    for (box, text, conf) in results:
        pts = np.array(box, dtype=np.float32)

        # Check if center is inside mask
        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()
        ix, iy = int(cx), int(cy)
        if iy < 0 or iy >= mask_bin.shape[0] or ix < 0 or ix >= mask_bin.shape[1]:
            continue
        if mask_bin[iy, ix] == 0:
            continue

        # Score: confidence + text similarity
        score = conf
        if source_text:
            s1 = text.lower().replace(" ", "")
            s2 = source_text.lower().replace(" ", "")
            if s1 and s2:
                common = sum(1 for c in s1 if c in s2)
                score += 0.5 * common / max(len(s1), len(s2))

        if score > best_score:
            best_score = score
            best = pts

    return best


def detect_on_cropped_region(reader, frame_rgb, mask_bin, source_text=None, pad_ratio=0.3):
    """Crop and enlarge the mask region, then run OCR for better detection of small text.

    This helps detect short/small text that gets missed in the full frame.
    """
    # Find mask bounding box
    ys, xs = np.where(mask_bin > 0)
    if len(ys) == 0:
        return None
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # Add padding
    h, w = frame_rgb.shape[:2]
    pad_h = int((y2 - y1) * pad_ratio)
    pad_w = int((x2 - x1) * pad_ratio)
    y1p = max(0, y1 - pad_h)
    y2p = min(h, y2 + pad_h)
    x1p = max(0, x1 - pad_w)
    x2p = min(w, x2 + pad_w)

    # Crop
    crop = frame_rgb[y1p:y2p, x1p:x2p]
    crop_mask = mask_bin[y1p:y2p, x1p:x2p]

    # Enlarge to at least 256px on the short side for better OCR
    ch, cw = crop.shape[:2]
    min_side = min(ch, cw)
    if min_side < 256:
        scale = 256.0 / min_side
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop_mask = cv2.resize(crop_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    else:
        scale = 1.0

    # Run OCR on enlarged crop
    results = reader.readtext(crop)
    if not results:
        return None

    best = None
    best_score = -1

    for (box, text, conf) in results:
        pts = np.array(box, dtype=np.float32)
        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()
        ix, iy = int(cx), int(cy)
        if iy < 0 or iy >= crop_mask.shape[0] or ix < 0 or ix >= crop_mask.shape[1]:
            continue
        if crop_mask[iy, ix] == 0:
            continue

        score = conf
        if source_text:
            s1 = text.lower().replace(" ", "")
            s2 = source_text.lower().replace(" ", "")
            if s1 and s2:
                common = sum(1 for c in s1 if c in s2)
                score += 0.5 * common / max(len(s1), len(s2))

        if score > best_score:
            best_score = score
            best = pts

    if best is None:
        return None

    # Map coordinates back to original frame
    best = best / scale
    best[:, 0] += x1p
    best[:, 1] += y1p

    return best


def detect_with_fallback(reader, frames_rgb, mask_frames_bin, source_text, max_tries=10):
    """Try detecting text in first N frames until we get a good detection.
    Falls back to cropped+enlarged detection if full-frame detection fails.
    """
    # Pass 1: full-frame detection
    for i in range(min(max_tries, len(frames_rgb))):
        box = detect_text_first_frame(reader, frames_rgb[i], mask_frames_bin[i], source_text)
        if box is not None:
            return box, i

    # Pass 2: crop mask region, enlarge, and retry
    for i in range(min(max_tries, len(frames_rgb))):
        box = detect_on_cropped_region(reader, frames_rgb[i], mask_frames_bin[i], source_text)
        if box is not None:
            return box, i

    return None, -1


# ---------------------------------------------------------------------------
# CoTracker2: track 4 corner points across video
# ---------------------------------------------------------------------------

_cotracker_model = None

def get_cotracker(device="cuda"):
    """Load CoTracker3 model (cached)."""
    global _cotracker_model
    if _cotracker_model is not None:
        return _cotracker_model

    from cotracker.predictor import CoTrackerPredictor

    # Download checkpoint to a file path
    ckpt_url = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
    ckpt_dir = os.path.join(os.path.expanduser("~"), ".cache", "cotracker")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "scaled_offline.pth")

    if not os.path.exists(ckpt_path):
        print("    Downloading CoTracker3 checkpoint...")
        state_dict = torch.hub.load_state_dict_from_url(ckpt_url, map_location="cpu")
        torch.save(state_dict, ckpt_path)

    _cotracker_model = CoTrackerPredictor(checkpoint=ckpt_path).to(device)
    return _cotracker_model


def track_points(video_tensor, init_points, init_frame_idx, device="cuda"):
    """Track 4 corner points across all video frames using CoTracker3.

    Args:
        video_tensor: (T, H, W, 3) uint8 numpy array
        init_points: (4, 2) numpy array of initial corner points
        init_frame_idx: which frame the points are detected in
        device: cuda or cpu

    Returns:
        tracked_points: (T, 4, 2) numpy array of tracked corner positions
        visibility: (T, 4) boolean array
    """
    model = get_cotracker(device)

    # Prepare video: (1, T, 3, H, W) float
    video = torch.from_numpy(video_tensor).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    # Prepare queries: (1, N, 3) where each query is (frame_idx, x, y)
    queries = torch.zeros(1, 4, 3)
    for i in range(4):
        queries[0, i, 0] = float(init_frame_idx)
        queries[0, i, 1] = float(init_points[i, 0])  # x
        queries[0, i, 2] = float(init_points[i, 1])  # y
    queries = queries.to(device)

    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, queries=queries)

    # pred_tracks: (1, T, 4, 2), pred_visibility: (1, T, 4)
    tracked = pred_tracks[0].cpu().numpy()     # (T, 4, 2)
    visible = pred_visibility[0].cpu().numpy()  # (T, 4)

    return tracked, visible


# ---------------------------------------------------------------------------
# Perspective glyph rendering (reused from render_glyph_ocr.py)
# ---------------------------------------------------------------------------

def render_text_horizontal(text, width, height, font_path):
    """Render text on a horizontal canvas, white text on black."""
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)

    font_size = max(8, height - 4)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    if text_w > width and text_w > 0:
        font_size = int(font_size * width / text_w * 0.9)
        font_size = max(8, font_size)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]

    text_h = bbox[3] - bbox[1]
    tx = (width - text_w) // 2
    ty = (height - text_h) // 2
    draw.text((tx, ty), text, fill=255, font=font)
    return np.array(img)


def render_glyph_perspective(target_text, dst_points, frame_h, frame_w, font_path):
    """Render target text warped to match the 4 corner points."""
    pts = np.array(dst_points, dtype=np.float32)

    w = int(np.linalg.norm(pts[1] - pts[0]))
    h = int(np.linalg.norm(pts[3] - pts[0]))
    if w < 5 or h < 5:
        return np.zeros((frame_h, frame_w), dtype=np.uint8)

    canvas_w, canvas_h = max(w * 2, 20), max(h * 2, 20)
    canvas = render_text_horizontal(target_text, canvas_w, canvas_h, font_path)

    src_pts = np.array(
        [[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_pts, pts)
    result = cv2.warpPerspective(canvas, M, (frame_w, frame_h))
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_video_frames(video_path):
    """Load all frames as RGB numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps


def load_mask_frames(mask_path, total_frames):
    """Load mask frames as binary arrays."""
    cap = cv2.VideoCapture(mask_path)
    masks = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        masks.append(mask_bin)
    cap.release()
    return masks


def process_one_sample(
    ocr, vid_id, video_path, mask_path,
    target_text, source_text, font_path, output_path,
    device="cuda",
):
    """Process one video: PaddleOCR detect → CoTracker track → render glyph."""

    # Load video and mask
    frames_rgb, fps = load_video_frames(video_path)
    total_frames = len(frames_rgb)
    masks_bin = load_mask_frames(mask_path, total_frames)
    frame_h, frame_w = frames_rgb[0].shape[:2]

    # Step 1: PaddleOCR detection on first frames (with fallback)
    init_box, init_frame = detect_with_fallback(
        ocr, frames_rgb, masks_bin, source_text, max_tries=10
    )

    if init_box is None:
        print(f"    WARN: No text detected in mask region for {vid_id}")
        return False

    print(f"    Detected in frame {init_frame}: box shape {init_box.shape}")

    # Step 2: CoTracker2 point tracking
    video_array = np.stack(frames_rgb)  # (T, H, W, 3)
    tracked_pts, visibility = track_points(
        video_array, init_box, init_frame, device=device
    )
    # tracked_pts: (T, 4, 2)

    # Step 3: Render glyph video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    for fi in range(total_frames):
        pts = tracked_pts[fi]  # (4, 2)
        vis = visibility[fi]   # (4,)

        # Render if at least 2 points visible (CoTracker still outputs
        # reasonable positions for "invisible" points near frame edges)
        if vis.sum() >= 2:
            # Clamp points to frame boundaries
            pts_clamped = pts.copy()
            pts_clamped[:, 0] = np.clip(pts_clamped[:, 0], 0, frame_w - 1)
            pts_clamped[:, 1] = np.clip(pts_clamped[:, 1], 0, frame_h - 1)

            glyph_frame = render_glyph_perspective(
                target_text, pts_clamped, frame_h, frame_w, font_path
            )
            glyph_bgr = cv2.cvtColor(
                cv2.merge([glyph_frame, glyph_frame, glyph_frame]),
                cv2.COLOR_RGB2BGR,
            )
        else:
            glyph_bgr = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        writer.write(glyph_bgr)

    writer.release()
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed/glyph_videos_tracked")
    parser.add_argument("--records", default="data/processed/parsed_records.json")
    parser.add_argument("--font_info_dir", default="data/processed/font_info")
    parser.add_argument("--sample_id", default=None, help="Process single sample")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.records) as f:
        records = json.load(f)

    if args.sample_id:
        records = [r for r in records if r["id"] == args.sample_id]
        if not records:
            print(f"Sample {args.sample_id} not found")
            return

    print(f"Initializing EasyOCR...")
    ocr = init_ocr(langs=("en", "ch_sim"))

    print(f"Processing {len(records)} samples...")
    success, fail = 0, 0

    for i, rec in enumerate(records):
        vid_id = rec["id"]
        target_text = rec["target_text"]
        source_text = rec["source_text"]

        video_path = os.path.join(args.raw_dir, rec["original_video"])
        mask_path = os.path.join(args.raw_dir, rec["mask_video"])
        output_path = os.path.join(args.output_dir, vid_id + ".mp4")

        if os.path.exists(output_path):
            success += 1
            continue

        # Font
        font_info_path = os.path.join(args.font_info_dir, vid_id + ".json")
        if os.path.exists(font_info_path):
            with open(font_info_path) as f:
                font_path = json.load(f)["resolved_font_path"]
        else:
            font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

        script = _detect_script(target_text)
        if script in _SCRIPT_FONTS:
            font_path = _SCRIPT_FONTS[script]

        print(f"  [{i+1}/{len(records)}] {vid_id}: \"{source_text}\" -> \"{target_text}\"")

        ok = process_one_sample(
            ocr, vid_id, video_path, mask_path,
            target_text, source_text, font_path, output_path,
            device=args.device,
        )

        if ok:
            success += 1
        else:
            fail += 1

    print(f"\nDone. Success: {success}, Failed: {fail}")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
