"""
Metric 4: VLM Quality Score
=============================
Uses a Vision-Language Model (qwen3-vl via ollama) to evaluate text editing
quality by asking the VLM to find and rate the target text in the edited frame.

The VLM sees the full frame, is told what text should appear, then:
1. Reports what text it actually sees
2. Rates the rendering quality 1-10

Uses qwen3-vl:32b-instruct via local ollama for best discrimination.
Inspired by Physics-IQ's MLLM evaluation approach.
"""

import base64
import json
import re
import cv2
import numpy as np
import requests


EVAL_PROMPT = '''In this video frame, the text "{target}" should appear somewhere.
Find it and rate how well it is rendered on a scale of 1-10:
1 = the target text is not present, text area is garbled or destroyed
3 = some characters are recognizable but distorted
5 = text is readable but has visible artifacts or style mismatch
8 = text is clear and mostly natural looking
10 = text is perfectly rendered, indistinguishable from real text
Reply in this exact format (two lines only):
Seen: <text you actually see>
Rating: <number>
/no_think'''


def _frame_to_base64(frame_bgr) -> str:
    """Convert a BGR frame to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


def _query_ollama(image_b64: str, prompt: str, model: str) -> str:
    """Send an image + prompt to ollama via chat API."""
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300},
        },
        timeout=180,
    )
    resp.raise_for_status()
    msg = resp.json()["message"]
    content = msg.get("content", "")
    if not content.strip():
        content = msg.get("thinking", "")
    return content


def _parse_response(response: str) -> dict:
    """Parse VLM response to extract seen text and rating."""
    lines = response.strip().split("\n")
    seen = ""
    rating = 0

    for line in lines:
        line = line.strip()
        if line.lower().startswith("seen:"):
            seen = line[5:].strip()
        elif line.lower().startswith("rating:"):
            try:
                rating = int(re.search(r'\d+', line).group())
                rating = max(1, min(10, rating))
            except (AttributeError, ValueError):
                rating = 0

    # Fallback: try to find a number anywhere if rating not found
    if rating == 0:
        numbers = re.findall(r'\b(\d+)\b', response)
        for n in numbers:
            n = int(n)
            if 1 <= n <= 10:
                rating = n
                break

    return {"seen_text": seen, "rating": rating}


def evaluate_vlm_quality(
    edited_video_path: str,
    target_text: str,
    model: str = "qwen3-vl:32b-instruct",
    n_frames: int = 3,
) -> dict:
    """Evaluate video editing quality using VLM.

    Shows the VLM the full edited frame, tells it what text should appear,
    and asks it to find and rate the text rendering quality.

    Args:
        edited_video_path: path to the edited video
        target_text: the text that should appear after editing
        model: ollama model name
        n_frames: number of frames to evaluate (first, middle, last)

    Returns:
        dict with vlm_score (1-10), seen_texts (what VLM read), frames_evaluated
    """
    cap = cv2.VideoCapture(edited_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return {"vlm_score": 0, "seen_texts": [], "frames_evaluated": 0}

    # Select frames: first, middle, last
    if total_frames <= n_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [0, total_frames // 2, total_frames - 1][:n_frames]

    ratings = []
    seen_texts = []

    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        img_b64 = _frame_to_base64(frame)
        prompt = EVAL_PROMPT.format(target=target_text)

        try:
            response = _query_ollama(img_b64, prompt, model=model)
            result = _parse_response(response)
            if result["rating"] > 0:
                ratings.append(result["rating"])
                seen_texts.append(result["seen_text"])
        except Exception as e:
            print(f"    VLM query failed for frame {fi}: {e}")

    cap.release()

    if not ratings:
        return {"vlm_score": 0, "seen_texts": [], "frames_evaluated": 0}

    return {
        "vlm_score": float(np.mean(ratings)),
        "seen_texts": seen_texts,
        "frames_evaluated": len(ratings),
    }
