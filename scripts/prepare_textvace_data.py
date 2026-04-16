"""
TextVACE Data Preparation Script
=================================
Processes raw data into training-ready format:
  1. Parse edit instructions → extract source/target text
  2. Font recognition via VLM (or fallback to default font)
  3. Render glyph condition videos
  4. Generate metadata.csv

Usage:
  conda run -n DiffSynth-Studio python scripts/prepare_textvace_data.py \
      --raw_dir data/raw --output_dir data/processed --step all

Steps can be run individually: parse, render_glyphs, metadata
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Step 1: Parse edit instructions
# ---------------------------------------------------------------------------

def parse_all_instructions(raw_dir: str) -> list[dict]:
    """Parse all edit instructions and pair with video/mask files."""
    inst_dir = os.path.join(raw_dir, "edit_instructions")
    orig_dir = os.path.join(raw_dir, "original_videos")
    edit_dir = os.path.join(raw_dir, "edited_videos")
    mask_dir = os.path.join(raw_dir, "text_masks")

    records = []
    for fname in sorted(os.listdir(inst_dir)):
        if not fname.endswith(".json") or fname == "manifest.json":
            continue

        vid_id = fname.replace(".json", "")
        inst_path = os.path.join(inst_dir, fname)

        with open(inst_path, "r") as f:
            data = json.load(f)
        instruction = data["instruction_en"]

        m = re.match(r"Change (.+) to (.+)", instruction)
        if not m:
            print(f"WARNING: Cannot parse instruction for {vid_id}: {instruction}")
            continue

        source_text = m.group(1)
        target_text = m.group(2)

        orig_path = os.path.join(orig_dir, vid_id + ".mp4")
        mask_path = os.path.join(mask_dir, vid_id + ".mp4")

        edited_path = os.path.join(edit_dir, vid_id + ".mp4")
        if not os.path.exists(edited_path):
            edited_path = os.path.join(edit_dir, vid_id + "_overlay.mp4")

        if not all(os.path.exists(p) for p in [orig_path, edited_path, mask_path]):
            print(f"WARNING: Missing files for {vid_id}")
            continue

        records.append({
            "id": vid_id,
            "original_video": os.path.relpath(orig_path, raw_dir),
            "edited_video": os.path.relpath(edited_path, raw_dir),
            "mask_video": os.path.relpath(mask_path, raw_dir),
            "source_text": source_text,
            "target_text": target_text,
            "instruction": instruction,
        })

    return records


# ---------------------------------------------------------------------------
# Step 2: Glyph rendering
# ---------------------------------------------------------------------------

def get_mask_bbox_per_frame(mask_video_path: str, num_frames: int = None) -> list[tuple]:
    """Extract bounding box of mask region for each frame.
    Returns list of (x, y, w, h) or None if no mask in that frame.
    """
    cap = cv2.VideoCapture(mask_video_path)
    bboxes = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if num_frames is not None and frame_idx >= num_frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(binary)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            bboxes.append((x, y, w, h))
        else:
            bboxes.append(None)
        frame_idx += 1
    cap.release()
    return bboxes


def _load_font(font_path: str, size: int):
    """Load a font, with fallback chain for robustness."""
    fallbacks = [
        font_path,
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in fallbacks:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
    return ImageFont.load_default()


def _detect_script(text: str) -> str:
    """Detect dominant script to choose appropriate font."""
    for ch in text:
        cp = ord(ch)
        # CJK Unified Ideographs + extensions
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            return "cjk"
        # Hangul
        if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            return "cjk"
        # Hiragana / Katakana
        if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
            return "cjk"
        # Cyrillic
        if 0x0400 <= cp <= 0x04FF:
            return "cyrillic"
        # Math symbols + arrows
        if 0x2200 <= cp <= 0x22FF or 0x2190 <= cp <= 0x21FF:
            return "symbol"
        # Roman numerals
        if 0x2160 <= cp <= 0x217F:
            return "symbol"
        # Enclosed alphanumerics (①②③)
        if 0x2460 <= cp <= 0x24FF:
            return "symbol"
        # Dingbats (✓✗✦)
        if 0x2700 <= cp <= 0x27BF:
            return "symbol"
        # Misc symbols (★☆♠♥)
        if 0x2600 <= cp <= 0x26FF:
            return "symbol"
        # Superscripts/subscripts
        if 0x2070 <= cp <= 0x209F:
            return "symbol"
    return "latin"


# Script-specific font overrides (DejaVu has best Unicode symbol coverage)
_SCRIPT_FONTS = {
    "cjk": "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "cyrillic": "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "symbol": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "math": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
}


def _resolve_font_for_text(text: str, font_path: str) -> str:
    """Choose font: script-specific for CJK/Cyrillic/math, VLM-suggested for Latin."""
    script = _detect_script(text)
    if script in _SCRIPT_FONTS:
        return _SCRIPT_FONTS[script]
    return font_path


def render_text_on_frame(
    text: str,
    frame_h: int,
    frame_w: int,
    bbox: tuple,
    font_path: str = None,
) -> np.ndarray:
    """Render white text on black background, positioned at bbox."""
    img = Image.new("RGB", (frame_w, frame_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    if bbox is None:
        return np.array(img)

    x, y, w, h = bbox

    # Override font based on script detection
    font_path = _resolve_font_for_text(text, font_path)

    # Find font size that fits the bbox
    font_size = max(8, h - 4)
    font = _load_font(font_path, font_size)

    # Measure text and adjust font size to fit width
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    if text_w > 0 and text_w > w:
        font_size = int(font_size * w / text_w * 0.95)
        font_size = max(8, font_size)
        font = _load_font(font_path, font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

    # Center text in bbox
    text_x = x + (w - text_w) // 2
    text_y = y + (h - text_h) // 2

    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def render_glyph_video(
    record: dict,
    raw_dir: str,
    output_dir: str,
    font_path: str = None,
) -> str:
    """Render a glyph condition video for one sample."""
    mask_path = os.path.join(raw_dir, record["mask_video"])
    orig_path = os.path.join(raw_dir, record["original_video"])
    target_text = record["target_text"]
    vid_id = record["id"]

    # Get video properties from original
    cap = cv2.VideoCapture(orig_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Get per-frame bboxes from mask
    bboxes = get_mask_bbox_per_frame(mask_path, total_frames)

    # Render each frame
    out_path = os.path.join(output_dir, vid_id + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

    for bbox in bboxes:
        frame = render_text_on_frame(target_text, frame_h, frame_w, bbox, font_path)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    return out_path


def render_all_glyphs(records: list[dict], raw_dir: str, output_dir: str, font_path: str = None):
    """Render glyph videos for all records."""
    os.makedirs(output_dir, exist_ok=True)
    for i, record in enumerate(records):
        out_path = render_glyph_video(record, raw_dir, output_dir, font_path)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Rendered glyph {i+1}/{len(records)}: {os.path.basename(out_path)}")
    print(f"  Done. {len(records)} glyph videos saved to {output_dir}")


# ---------------------------------------------------------------------------
# Step 3: Generate metadata.csv
# ---------------------------------------------------------------------------

def generate_metadata(records: list[dict], raw_dir: str, glyph_dir: str, output_csv: str):
    """Generate metadata.csv for training.

    Column mapping for VACE training:
      video         = edited_video (ground truth, training target)
      vace_video    = original_video (condition input)
      vace_video_mask = mask_video (text region mask)
      glyph_video   = rendered glyph video (new condition)
      prompt        = edit instruction text
    """
    rows = []
    for record in records:
        glyph_path = os.path.join(
            os.path.relpath(glyph_dir, os.path.dirname(output_csv)),
            record["id"] + ".mp4"
        )
        rows.append({
            "video": record["edited_video"],
            "vace_video": record["original_video"],
            "vace_video_mask": record["mask_video"],
            "glyph_video": glyph_path,
            "prompt": record["instruction"],
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "vace_video", "vace_video_mask", "glyph_video", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Metadata saved: {output_csv} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TextVACE data preparation")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw data")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--font_path", type=str, default=None, help="Path to .ttf font file (default: DejaVu Sans Bold)")
    parser.add_argument("--step", type=str, default="all", choices=["all", "parse", "render_glyphs", "metadata"],
                        help="Which step to run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    glyph_dir = os.path.join(args.output_dir, "glyph_videos")

    # Step 1: Parse
    print("=" * 60)
    print("Step 1: Parsing edit instructions...")
    records = parse_all_instructions(args.raw_dir)
    print(f"  Parsed {len(records)} records")

    # Save parsed records for reference
    parsed_path = os.path.join(args.output_dir, "parsed_records.json")
    with open(parsed_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {parsed_path}")

    if args.step == "parse":
        return

    # Step 2: Render glyphs
    print("=" * 60)
    print("Step 2: Rendering glyph condition videos...")
    render_all_glyphs(records, args.raw_dir, glyph_dir, args.font_path)

    if args.step == "render_glyphs":
        return

    # Step 3: Metadata
    print("=" * 60)
    print("Step 3: Generating metadata.csv...")
    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    generate_metadata(records, args.raw_dir, glyph_dir, metadata_path)

    print("=" * 60)
    print("All done!")
    print(f"  Glyph videos: {glyph_dir}/")
    print(f"  Metadata:     {metadata_path}")


if __name__ == "__main__":
    main()
