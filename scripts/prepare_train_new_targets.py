"""
Generate new target texts for training data using local VLM.
For each training sample, ask the VLM to generate a different replacement text
(different from the original target_text used in training).
Then render new glyph videos and create metadata for inference.
"""

import os
import sys
import cv2
import json
import glob
import time
import numpy as np
import base64
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
import subprocess

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_GLYPH_DIR = os.path.join(DATA_DIR, "processed/glyph_videos_new_targets")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")

OLLAMA_MODEL = "gemma3:27b"
OLLAMA_URL = "http://127.0.0.1:11434"

FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


def call_vlm_new_target(source_text, original_target, cropped_img):
    """Ask VLM to generate a NEW target text different from the original."""
    _, crop_buf = cv2.imencode('.png', cropped_img)
    crop_b64 = base64.b64encode(crop_buf).decode('utf-8')

    prompt = f"""The image shows text from a video frame. The original text reads: "{source_text}"

In a previous edit, this was changed to: "{original_target}"

Now I need a DIFFERENT replacement text. Requirements:
- Must be DIFFERENT from both "{source_text}" and "{original_target}"
- Similar length to the source text (within 2x characters)
- Same language/script as the source
- Makes sense as a replacement in the same context

Respond in this exact JSON format only, no markdown:
{{"new_target_text": "...", "instruction": "Change {source_text} to ..."}}"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [crop_b64],
            "stream": False,
            "options": {"temperature": 0.7},
        },
        timeout=120,
    )
    response.raise_for_status()
    text = response.json()["response"].strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return json.loads(text)


def find_font(target_size=40):
    for font_path in FALLBACK_FONTS:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, target_size)
    return ImageFont.load_default()


def render_glyph_frame(target_text, mask_bin, frame_shape):
    h, w = frame_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    coords = np.where(mask_bin > 0)
    if len(coords[0]) == 0:
        return canvas
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    box_h = y_max - y_min
    box_w = x_max - x_min

    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)

    low, high = 8, min(box_h, 200)
    best_font = find_font(low)
    while low <= high:
        mid = (low + high) // 2
        test_font = find_font(mid)
        bbox = draw.textbbox((0, 0), target_text, font=test_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= box_w * 0.95 and th <= box_h * 0.95:
            best_font = test_font
            low = mid + 1
        else:
            high = mid - 1

    bbox = draw.textbbox((0, 0), target_text, font=best_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_x = x_min + (box_w - tw) // 2
    text_y = y_min + (box_h - th) // 2

    draw.text((text_x, text_y), target_text, fill=(255, 255, 255), font=best_font)

    result = np.array(pil_canvas)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expanded_mask = cv2.dilate(mask_bin, kernel, iterations=2)
    result[expanded_mask == 0] = 0

    return result


def write_video_h264(frames, output_path, fps, w, h):
    cmd = [
        FFMPEG, '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    return proc.returncode == 0


def create_glyph_video(mask_path, target_text, output_path):
    cap = cv2.VideoCapture(mask_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    masks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        masks.append(binary)
    cap.release()

    if not masks:
        return False

    frames = []
    for mask in masks:
        glyph_frame = render_glyph_frame(target_text, mask, (h, w))
        frames.append(glyph_frame)

    return write_video_h264(frames, output_path, fps, w, h)


def extract_masked_crop(video_path, mask_path):
    """Extract cropped region around mask from first frame."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    mask_cap = cv2.VideoCapture(mask_path)
    ret, mask_frame = mask_cap.read()
    mask_cap.release()
    if not ret:
        return None

    mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask_gray, 50, 255, cv2.THRESH_BINARY)

    coords = np.where(mask_bin > 0)
    if len(coords[0]) == 0:
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    h, w = frame.shape[:2]
    pad_y = int((y_max - y_min) * 0.3)
    pad_x = int((x_max - x_min) * 0.3)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)

    return frame[y_min:y_max, x_min:x_max]


def main():
    # Load existing training records
    with open(os.path.join(DATA_DIR, "processed/parsed_records.json")) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} training records")

    os.makedirs(OUTPUT_GLYPH_DIR, exist_ok=True)

    new_records = []
    errors = []

    for idx, rec in enumerate(records):
        video_id = rec["id"]
        source_text = rec["source_text"]
        original_target = rec["target_text"]

        print(f"\n[{idx+1}/{len(records)}] {video_id}: '{source_text}' (was -> '{original_target}')")

        glyph_out = os.path.join(OUTPUT_GLYPH_DIR, f"{video_id}.mp4")
        if os.path.exists(glyph_out):
            print(f"  SKIP: Already processed")
            continue

        # Get cropped image of text region
        video_path = os.path.join(RAW_DIR, rec["original_video"])
        mask_path = os.path.join(RAW_DIR, rec["mask_video"])

        cropped = extract_masked_crop(video_path, mask_path)
        if cropped is None:
            print(f"  SKIP: Could not extract crop")
            errors.append({"id": video_id, "error": "empty crop"})
            continue

        # Call VLM for new target
        try:
            result = call_vlm_new_target(source_text, original_target, cropped)
            new_target = result["new_target_text"]
            instruction = result["instruction"]
            print(f"  New target: '{new_target}' ({instruction})")
        except Exception as e:
            print(f"  ERROR VLM: {e}")
            errors.append({"id": video_id, "error": str(e)})
            continue

        # Render new glyph video
        try:
            success = create_glyph_video(mask_path, new_target, glyph_out)
            if success:
                print(f"  Glyph saved: {glyph_out}")
            else:
                print(f"  ERROR creating glyph")
                errors.append({"id": video_id, "error": "glyph failed"})
                continue
        except Exception as e:
            print(f"  ERROR glyph: {e}")
            errors.append({"id": video_id, "error": str(e)})
            continue

        new_records.append({
            "id": video_id,
            "source_text": source_text,
            "original_target": original_target,
            "new_target_text": new_target,
            "instruction": instruction,
        })

        time.sleep(0.5)

        if (idx + 1) % 10 == 0:
            with open(os.path.join(OUTPUT_DIR, "new_target_records.json"), "w") as f:
                json.dump(new_records, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(new_records)} records")

    # Save final
    with open(os.path.join(OUTPUT_DIR, "new_target_records.json"), "w") as f:
        json.dump(new_records, f, indent=2, ensure_ascii=False)

    if errors:
        with open(os.path.join(OUTPUT_DIR, "new_target_errors.json"), "w") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)

    # Generate metadata CSV for inference
    csv_path = os.path.join(DATA_DIR, "metadata_new_targets.csv")
    with open(csv_path, "w") as f:
        f.write("video,vace_video,vace_video_mask,glyph_video,prompt\n")
        for rec in new_records:
            vid = rec["id"]
            # Use original video as input, new glyph as guide
            f.write(f"raw/original_videos/{vid}.mp4,"
                    f"raw/original_videos/{vid}.mp4,"
                    f"raw/text_masks/{vid}.mp4,"
                    f"processed/glyph_videos_new_targets/{vid}.mp4,"
                    f"{rec['instruction']}\n")

    print(f"\nDone! {len(new_records)} new targets, {len(errors)} errors")
    print(f"Metadata: {csv_path}")
    print(f"Records: {os.path.join(OUTPUT_DIR, 'new_target_records.json')}")


if __name__ == "__main__":
    main()
