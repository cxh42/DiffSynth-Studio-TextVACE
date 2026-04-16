"""
Font Recognition via VLM (ollama + qwen3-vl)
=============================================
For each sample, extract the text region from the first frame,
send it to a local VLM to identify the closest standard font,
and save results to data/processed/font_info/.

Usage:
  conda run -n DiffSynth-Studio python scripts/recognize_fonts.py \
      --raw_dir data/raw --output_dir data/processed \
      --model qwen3-vl:8b-instruct
"""

import argparse
import base64
import json
import os
import sys
import time

import cv2
import numpy as np
import requests


# Map VLM-returned font names to system font paths.
# Liberation fonts are metric-compatible replacements:
#   Arial -> Liberation Sans, Times New Roman -> Liberation Serif, Courier New -> Liberation Mono
FONT_MAP = {
    # === Sans-serif ===
    "arial": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "arial bold": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "helvetica": "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
    "helvetica bold": "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
    "verdana": "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "calibri": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "trebuchet ms": "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "sans-serif": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "sans-serif bold": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    # Condensed/Narrow sans
    "arial narrow": "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
    "condensed": "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Regular.otf",
    "condensed bold": "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Bold.otf",
    # === Serif ===
    "times new roman": "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
    "times new roman bold": "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf",
    "times": "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
    "georgia": "/usr/share/fonts/opentype/urw-base35/P052-Roman.otf",
    "palatino": "/usr/share/fonts/opentype/urw-base35/P052-Roman.otf",
    "book antiqua": "/usr/share/fonts/opentype/urw-base35/P052-Roman.otf",
    "bookman": "/usr/share/fonts/opentype/urw-base35/URWBookman-Light.otf",
    "bookman bold": "/usr/share/fonts/opentype/urw-base35/URWBookman-Demi.otf",
    "century schoolbook": "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf",
    "serif": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "serif bold": "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    # === Monospace ===
    "courier": "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf",
    "courier new": "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf",
    "courier new bold": "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Bold.otf",
    "monospace": "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    # === Display / Bold / Impact ===
    "impact": "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
    "impact bold": "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
    "franklin gothic": "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
    # === Script / Handwritten / Cursive ===
    "script": "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    "script mt bold": "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    "cursive": "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    "handwritten": "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    "calligraphy": "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    "zapf chancery": "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    "comic sans ms": "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    "comic sans": "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    "brush script": "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    # === Gothic / Avant Garde ===
    "avant garde": "/usr/share/fonts/opentype/urw-base35/URWGothic-Book.otf",
    "avant garde bold": "/usr/share/fonts/opentype/urw-base35/URWGothic-Demi.otf",
    "century gothic": "/usr/share/fonts/opentype/urw-base35/URWGothic-Book.otf",
    "futura": "/usr/share/fonts/opentype/urw-base35/URWGothic-Book.otf",
    # === Italic variants ===
    "italic": "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    "sans-serif italic": "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    "serif italic": "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
    # === Symbols / Dingbats ===
    "symbol": "/usr/share/fonts/opentype/urw-base35/StandardSymbolsPS.otf",
    "dingbats": "/usr/share/fonts/opentype/urw-base35/D050000L.otf",
    # === CJK ===
    "simhei": "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "microsoft yahei": "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "malgun gothic": "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "noto sans cjk": "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "songti": "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "simsun": "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
}

DEFAULT_FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

# Script-specific fonts for characters that Latin fonts can't render
SCRIPT_FONTS = {
    "cjk": "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "cyrillic": "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "latin_extended": "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "math": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
}


def detect_script(text: str) -> str:
    """Detect the dominant script in the text to pick an appropriate font."""
    for ch in text:
        cp = ord(ch)
        # CJK Unified Ideographs + CJK extensions + Hangul + Kana
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:  # CJK
            return "cjk"
        if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:  # Hangul
            return "cjk"
        if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:  # Hiragana/Katakana
            return "cjk"
        if 0x0400 <= cp <= 0x04FF:  # Cyrillic
            return "cyrillic"
        if 0x2200 <= cp <= 0x22FF or 0x2190 <= cp <= 0x21FF:  # Math symbols
            return "math"
    return "latin"


def get_font_for_text(text: str, vlm_font_path: str) -> str:
    """Choose the best font: use VLM-suggested font for Latin, script-specific for others."""
    script = detect_script(text)
    if script in SCRIPT_FONTS:
        return SCRIPT_FONTS[script]
    return vlm_font_path


def resolve_font_path(font_name: str) -> str:
    """Map a VLM-returned font name to an available system font path."""
    key = font_name.strip().lower()
    # Direct match
    if key in FONT_MAP:
        return FONT_MAP[key]
    # Partial match (try both directions)
    for k, v in FONT_MAP.items():
        if k in key or key in k:
            return v
    # Style keyword fallback
    if "script" in key or "handwrit" in key or "cursive" in key or "brush" in key or "calligraph" in key:
        return "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf"
    if "gothic" in key or "futura" in key or "avant" in key:
        return "/usr/share/fonts/opentype/urw-base35/URWGothic-Book.otf"
    if "condensed" in key or "narrow" in key:
        return "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf"
    if "italic" in key:
        return "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"
    if "bold" in key and "serif" in key and "sans" not in key:
        return "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf"
    if "bold" in key:
        return "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    if "serif" in key and "sans" not in key:
        return "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf"
    if "mono" in key:
        return "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf"
    return DEFAULT_FONT


def extract_text_crop(orig_video_path: str, mask_video_path: str, pad: int = 20) -> bytes:
    """Extract the text region from the first frame as PNG bytes."""
    cap = cv2.VideoCapture(orig_video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    cap2 = cv2.VideoCapture(mask_video_path)
    ret2, mask = cap2.read()
    cap2.release()
    if not ret2:
        return None

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)
    crop = frame[y1:y2, x1:x2]

    _, png_bytes = cv2.imencode(".png", crop)
    return png_bytes.tobytes()


def query_vlm(image_bytes: bytes, source_text: str, model: str = "qwen3-vl:8b-instruct",
              ollama_url: str = "http://localhost:11434") -> str:
    """Query VLM for font recognition."""
    b64 = base64.b64encode(image_bytes).decode()

    prompt = (
        f'This image shows the text "{source_text}" from a video frame. '
        f'Identify the font style. Consider these categories:\n'
        f'- Sans-serif regular (clean, no serifs): Arial, Helvetica, Verdana, Calibri\n'
        f'- Sans-serif bold (thick, heavy): Arial Bold, Helvetica Bold, Impact, Franklin Gothic\n'
        f'- Sans-serif condensed (tall and narrow): Arial Narrow, Condensed Bold\n'
        f'- Serif regular (with serifs): Times New Roman, Georgia, Palatino, Bookman\n'
        f'- Serif bold: Times New Roman Bold, Georgia Bold\n'
        f'- Monospace (fixed-width): Courier New, Courier New Bold\n'
        f'- Script/Handwritten (cursive, flowing): Script, Brush Script, Cursive, Handwritten\n'
        f'- Gothic/Geometric (round, modern): Century Gothic, Futura, Avant Garde\n'
        f'- Italic (slanted): Italic, Sans-serif Italic\n'
        f'Reply with ONLY the font name from the examples above. No explanation.'
    )

    try:
        resp = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 30},
            },
            timeout=120,
        )
        data = resp.json()
        content = data.get("message", {}).get("content", "").strip()
        # Remove thinking tags if present (qwen3 sometimes wraps in <think>)
        if "<think>" in content:
            # Extract text after </think>
            parts = content.split("</think>")
            content = parts[-1].strip() if len(parts) > 1 else content
        # Clean: take first line only
        content = content.split("\n")[0].strip()
        return content
    except Exception as e:
        print(f"  VLM error: {e}")
        return "Arial Bold"


def main():
    parser = argparse.ArgumentParser(description="Font recognition via VLM")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--model", type=str, default="qwen3-vl:8b-instruct")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434")
    args = parser.parse_args()

    # Load parsed records
    parsed_path = os.path.join(args.output_dir, "parsed_records.json")
    if not os.path.exists(parsed_path):
        print(f"ERROR: {parsed_path} not found. Run prepare_textvace_data.py first.")
        sys.exit(1)

    with open(parsed_path) as f:
        records = json.load(f)

    font_dir = os.path.join(args.output_dir, "font_info")
    os.makedirs(font_dir, exist_ok=True)

    # Check which are already done
    done = set()
    for fname in os.listdir(font_dir):
        if fname.endswith(".json"):
            done.add(fname.replace(".json", ""))

    remaining = [r for r in records if r["id"] not in done]
    print(f"Total: {len(records)}, already done: {len(done)}, remaining: {len(remaining)}")

    for i, record in enumerate(remaining):
        vid_id = record["id"]
        orig_path = os.path.join(args.raw_dir, record["original_video"])
        mask_path = os.path.join(args.raw_dir, record["mask_video"])

        crop_bytes = extract_text_crop(orig_path, mask_path)
        if crop_bytes is None:
            print(f"  [{i+1}/{len(remaining)}] {vid_id}: SKIP (no crop)")
            font_name = "Arial Bold"
        else:
            font_name = query_vlm(crop_bytes, record["source_text"], args.model, args.ollama_url)

        font_path = resolve_font_path(font_name)

        result = {
            "id": vid_id,
            "vlm_font_name": font_name,
            "resolved_font_path": font_path,
        }

        out_path = os.path.join(font_dir, vid_id + ".json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(remaining)}] {vid_id}: \"{font_name}\" -> {os.path.basename(font_path)}")

    print(f"Done. Font info saved to {font_dir}/")

    # Summary
    fonts_counter = {}
    for fname in os.listdir(font_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(font_dir, fname)) as f:
            info = json.load(f)
        fn = info["vlm_font_name"]
        fonts_counter[fn] = fonts_counter.get(fn, 0) + 1

    print("\nFont distribution:")
    for fn, count in sorted(fonts_counter.items(), key=lambda x: -x[1]):
        print(f"  {fn}: {count}")


if __name__ == "__main__":
    main()
