"""
Prepare inference data from inference_raw/ for novel (unseen) videos.
1. Filter out training samples
2. Dilate masks to match training mask coverage
3. Use VLM to recognize text in masked regions
4. Generate novel target text and render glyph videos

Usage:
  conda run -n DiffSynth-Studio python scripts/prepare_inference_data.py
"""

import base64
import cv2
import json
import numpy as np
import os
import random
import requests
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.prepare_textvace_data import (
    get_mask_bbox_per_frame, render_text_on_frame,
    _detect_script, _SCRIPT_FONTS, _load_font,
)


def get_training_ids(raw_dir="data/raw"):
    return set(f.replace(".mp4", "") for f in os.listdir(os.path.join(raw_dir, "original_videos")))


def get_novel_ids(infer_dir="data/inference_raw", train_ids=None):
    all_ids = set(f.replace(".mp4", "") for f in os.listdir(os.path.join(infer_dir, "target_video")))
    if train_ids:
        return sorted(all_ids - train_ids)
    return sorted(all_ids)


def find_mask_file(vid_id, mask_dir):
    for f in os.listdir(mask_dir):
        if f.startswith(vid_id):
            return os.path.join(mask_dir, f)
    return None


def dilate_mask_video(mask_path, output_path, kernel_size=25, iterations=3):
    """Dilate mask to cover more area around text, matching training mask coverage."""
    cap = cv2.VideoCapture(mask_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(binary, kernel, iterations=iterations)
        frame_out = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        writer.write(frame_out)

    cap.release()
    writer.release()


def extract_text_region_frame(video_path, mask_path, pad=10):
    """Extract first frame's text region for VLM recognition."""
    cap_v = cv2.VideoCapture(video_path)
    ret_v, frame = cap_v.read()
    cap_v.release()

    cap_m = cv2.VideoCapture(mask_path)
    ret_m, mask = cap_m.read()
    cap_m.release()

    if not ret_v or not ret_m:
        return None

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    crop = frame[y1:y2, x1:x2]

    _, png = cv2.imencode(".png", crop)
    return png.tobytes()


def vlm_recognize_and_generate(image_bytes, model="qwen3-vl:8b-instruct", url="http://localhost:11434"):
    """Ask VLM to read the text AND suggest a replacement."""
    b64 = base64.b64encode(image_bytes).decode()

    prompt = (
        "Look at this image cropped from a video. "
        "1) What text is shown in this image? Read it exactly. "
        "2) Suggest a plausible replacement text of similar length. "
        "Reply in this exact format, nothing else:\n"
        "ORIGINAL: <the text you read>\n"
        "REPLACEMENT: <your suggested replacement>"
    )

    try:
        resp = requests.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 100},
            },
            timeout=120,
        )
        content = resp.json().get("message", {}).get("content", "").strip()

        # Remove thinking tags
        if "<think>" in content:
            content = content.split("</think>")[-1].strip()

        original = ""
        replacement = ""
        for line in content.split("\n"):
            line = line.strip()
            if line.upper().startswith("ORIGINAL:"):
                original = line.split(":", 1)[1].strip()
            elif line.upper().startswith("REPLACEMENT:"):
                replacement = line.split(":", 1)[1].strip()

        return original, replacement
    except Exception as e:
        print(f"  VLM error: {e}")
        return "", ""


def render_glyph_video(target_text, mask_path, video_path, font_path, output_path):
    """Render glyph video for target text."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    bboxes = get_mask_bbox_per_frame(mask_path, total)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for bbox in bboxes:
        frame = render_text_on_frame(target_text, h, w, bbox, font_path)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    infer_dir = "data/inference_raw"
    output_dir = "data/inference_processed"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "dilated_masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "glyph_videos"), exist_ok=True)

    train_ids = get_training_ids()
    novel_ids = get_novel_ids(infer_dir, train_ids)
    print(f"Novel (unseen) samples: {len(novel_ids)}")

    # Pick a diverse subset for inference (don't need all 240)
    random.seed(42)
    selected = random.sample(novel_ids, min(20, len(novel_ids)))
    print(f"Selected for inference: {len(selected)}")

    records = []
    for i, vid_id in enumerate(selected):
        video_path = os.path.join(infer_dir, "target_video", vid_id + ".mp4")
        mask_file = find_mask_file(vid_id, os.path.join(infer_dir, "mask_video"))

        if not mask_file or not os.path.exists(video_path):
            print(f"  [{i+1}] {vid_id}: SKIP (missing files)")
            continue

        print(f"  [{i+1}/{len(selected)}] {vid_id}")

        # 1. Dilate mask
        dilated_path = os.path.join(output_dir, "dilated_masks", vid_id + ".mp4")
        dilate_mask_video(mask_file, dilated_path)

        # 2. VLM recognize text
        crop_bytes = extract_text_region_frame(video_path, mask_file)
        if crop_bytes is None:
            print(f"    SKIP: no text region found")
            continue

        original_text, replacement_text = vlm_recognize_and_generate(crop_bytes)
        if not original_text or not replacement_text:
            print(f"    SKIP: VLM failed")
            continue

        # Clean up quotes
        original_text = original_text.strip('"\'')
        replacement_text = replacement_text.strip('"\'')

        print(f"    VLM: \"{original_text}\" -> \"{replacement_text}\"")

        # 3. Detect font and render glyph
        script = _detect_script(replacement_text)
        if script in _SCRIPT_FONTS:
            font_path = _SCRIPT_FONTS[script]
        else:
            font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

        glyph_path = os.path.join(output_dir, "glyph_videos", vid_id + ".mp4")
        render_glyph_video(replacement_text, dilated_path, video_path, font_path, glyph_path)

        prompt = f"Change {original_text} to {replacement_text}"
        records.append({
            "id": vid_id,
            "video_path": video_path,
            "mask_path": dilated_path,
            "glyph_path": glyph_path,
            "original_text": original_text,
            "replacement_text": replacement_text,
            "prompt": prompt,
        })

    # Save records
    out_json = os.path.join(output_dir, "inference_records.json")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(records)} records to {out_json}")


if __name__ == "__main__":
    main()
