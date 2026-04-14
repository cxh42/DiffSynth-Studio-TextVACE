"""
OCR-Driven Glyph Video Rendering
==================================
Uses EasyOCR to detect text geometry in each frame of the original video,
then renders the target text at the exact same position/size/angle/perspective.

Usage:
  conda run -n DiffSynth-Studio python scripts/render_glyph_ocr.py \
      --raw_dir data/raw --output_dir data/processed/glyph_videos_v2
"""

import argparse
import json
import os
import sys

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.prepare_textvace_data import _detect_script, _SCRIPT_FONTS


# ---------------------------------------------------------------------------
# OCR text detection
# ---------------------------------------------------------------------------

def detect_text_in_frame(reader, frame_rgb):
    """Run OCR on a frame, return list of (box_4pts, text, conf)."""
    results = reader.readtext(frame_rgb)
    return results


def find_target_detection(detections, mask_binary, source_text=None):
    """Find the OCR detection that falls inside the mask region.
    If source_text is provided, also match by text similarity.
    Returns the best matching (box_4pts, text, conf) or None.
    """
    best = None
    best_score = -1

    for (box, text, conf) in detections:
        pts = np.array(box)
        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()

        # Check if center is inside mask
        ix, iy = int(cx), int(cy)
        if iy < 0 or iy >= mask_binary.shape[0] or ix < 0 or ix >= mask_binary.shape[1]:
            continue
        if mask_binary[iy, ix] == 0:
            continue

        # Score: confidence + text similarity bonus
        score = conf
        if source_text:
            # Simple similarity: ratio of matching characters
            s1 = text.lower().replace(" ", "")
            s2 = source_text.lower().replace(" ", "")
            if s1 and s2:
                common = sum(1 for c in s1 if c in s2)
                score += 0.5 * common / max(len(s1), len(s2))

        if score > best_score:
            best_score = score
            best = (pts.tolist(), text, conf)

    return best


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def smooth_boxes(boxes_per_frame, alpha=0.7):
    """Smooth 4-corner boxes across frames to reduce jitter.
    boxes_per_frame: list of (4x2 array or None) per frame.
    Returns smoothed list.
    """
    smoothed = []
    prev = None
    for box in boxes_per_frame:
        if box is None:
            smoothed.append(prev)  # carry forward
            continue
        box = np.array(box, dtype=np.float64)
        if prev is None:
            prev = box
        else:
            prev = alpha * box + (1 - alpha) * prev
        smoothed.append(prev.copy())

    # Backward pass to fill leading Nones
    for i in range(len(smoothed) - 2, -1, -1):
        if smoothed[i] is None and smoothed[i + 1] is not None:
            smoothed[i] = smoothed[i + 1]

    return smoothed


def interpolate_missing(boxes_per_frame):
    """Fill None entries by linear interpolation from neighbors."""
    n = len(boxes_per_frame)
    result = list(boxes_per_frame)

    # Find segments of Nones and interpolate
    i = 0
    while i < n:
        if result[i] is None:
            # Find start and end of None segment
            start = i
            while i < n and result[i] is None:
                i += 1
            end = i  # first non-None after segment

            # Get boundary boxes for interpolation
            before = result[start - 1] if start > 0 else None
            after = result[end] if end < n else None

            if before is not None and after is not None:
                before = np.array(before)
                after = np.array(after)
                for j in range(start, end):
                    t = (j - start + 1) / (end - start + 1)
                    result[j] = (before * (1 - t) + after * t).tolist()
            elif before is not None:
                for j in range(start, end):
                    result[j] = before.tolist() if isinstance(before, np.ndarray) else before
            elif after is not None:
                for j in range(start, end):
                    result[j] = after.tolist() if isinstance(after, np.ndarray) else after
        else:
            i += 1

    return result


# ---------------------------------------------------------------------------
# Perspective glyph rendering
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

    # Adjust font size to fit width
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
    """Render target text warped to match the 4 corner points from OCR detection."""
    pts = np.array(dst_points, dtype=np.float32)

    # Compute text box dimensions
    w = int(np.linalg.norm(pts[1] - pts[0]))
    h = int(np.linalg.norm(pts[3] - pts[0]))
    if w < 5 or h < 5:
        return np.zeros((frame_h, frame_w), dtype=np.uint8)

    # Render on horizontal canvas (2x resolution for quality)
    canvas_w, canvas_h = max(w * 2, 20), max(h * 2, 20)
    canvas = render_text_horizontal(target_text, canvas_w, canvas_h, font_path)

    # Perspective transform: horizontal canvas → detected 4 corners
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

def process_one_sample(
    reader,
    vid_id,
    video_path,
    mask_path,
    target_text,
    source_text,
    font_path,
    output_path,
    sample_interval=4,
):
    """Process one video: OCR detect → track → render glyph video."""
    cap_v = cv2.VideoCapture(video_path)
    cap_m = cv2.VideoCapture(mask_path)
    fps = cap_v.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))

    # Step 1: OCR detect on sampled frames, find text box in mask region
    raw_boxes = [None] * total_frames
    for fi in range(total_frames):
        ret_v, frame = cap_v.read()
        ret_m, mask_frame = cap_m.read()
        if not ret_v or not ret_m:
            break

        # Only run OCR every N frames (expensive)
        if fi % sample_interval != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

        detections = detect_text_in_frame(reader, frame_rgb)
        match = find_target_detection(detections, mask_bin, source_text)
        if match:
            raw_boxes[fi] = match[0]  # 4 corner points

    cap_v.release()
    cap_m.release()

    # Step 2: Interpolate missing frames + smooth
    raw_boxes = interpolate_missing(raw_boxes)
    smoothed_boxes = smooth_boxes(raw_boxes, alpha=0.7)

    # Count valid detections
    valid = sum(1 for b in smoothed_boxes if b is not None)
    if valid == 0:
        # Fallback: use mask bounding box
        return False

    # Step 3: Render glyph video with perspective transform
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    for fi in range(total_frames):
        box = smoothed_boxes[fi] if fi < len(smoothed_boxes) else None
        if box is not None:
            glyph_frame = render_glyph_perspective(
                target_text, box, frame_h, frame_w, font_path
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed/glyph_videos_v2")
    parser.add_argument("--records", default="data/processed/parsed_records.json")
    parser.add_argument("--font_info_dir", default="data/processed/font_info")
    parser.add_argument("--sample_interval", type=int, default=4,
                        help="Run OCR every N frames (4 = every 4th frame)")
    parser.add_argument("--langs", default="en,ch_sim",
                        help="EasyOCR languages")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.records) as f:
        records = json.load(f)

    print(f"Initializing EasyOCR ({args.langs})...")
    langs = args.langs.split(",")
    reader = easyocr.Reader(langs, gpu=True)

    success, fail = 0, 0
    for i, rec in enumerate(records):
        vid_id = rec["id"]
        target_text = rec["target_text"]
        source_text = rec["source_text"]

        video_path = os.path.join(args.raw_dir, rec["original_video"])
        mask_path = os.path.join(args.raw_dir, rec["mask_video"])
        output_path = os.path.join(args.output_dir, vid_id + ".mp4")

        # Font
        font_info_path = os.path.join(args.font_info_dir, vid_id + ".json")
        if os.path.exists(font_info_path):
            with open(font_info_path) as f:
                font_path = json.load(f)["resolved_font_path"]
        else:
            font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

        # Script detection override
        script = _detect_script(target_text)
        if script in _SCRIPT_FONTS:
            font_path = _SCRIPT_FONTS[script]

        ok = process_one_sample(
            reader, vid_id, video_path, mask_path,
            target_text, source_text, font_path, output_path,
            sample_interval=args.sample_interval,
        )

        if ok:
            success += 1
        else:
            fail += 1

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(records)}] {vid_id}: {'OK' if ok else 'FALLBACK'} "
                  f"(\"{source_text}\" -> \"{target_text}\")")

    print(f"\nDone. Success: {success}, Fallback: {fail}")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
