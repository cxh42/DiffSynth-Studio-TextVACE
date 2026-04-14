#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import difflib
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageOps


SYSTEM_PROMPT = (
    "You extract the minimal text replacement instruction between an original video and "
    "its edited version. Focus only on the masked text region. Return strict JSON only."
)

USER_PROMPT = """You receive 3 images in order.
Image 1 is the context sheet:
- left: original frame with a red box
- middle: edited frame with a red box
- right: mask frame

Image 2 is the zoomed crop sheet:
- top row: original masked region across time
- bottom row: edited masked region across time

Image 3 is the focus sheet:
- top row: original pixels inside the mask only
- bottom row: edited pixels inside the same masked area only

Task:
1. Read only the text inside the mask. Ignore neighboring text outside the mask, even if it appears in the crop.
2. Determine the exact source_text before editing and target_text after editing.
3. If the masked text spans multiple lines, join the visible lines with a single space in reading order.

Output JSON fields:
- source_text
- target_text
- confidence

Rules:
- The masked region is the edited region, so source_text and target_text should normally be different.
- If your first reading makes them identical, look again and prefer the literal changed token inside the mask rather than surrounding context.
- If the edited text is an overlay or sticker, read the edited overlay rather than the original background text.
- Preserve the literal text exactly, including numbers and punctuation.
- If you are uncertain, make your best literal reading and lower confidence.
- Do not return markdown or extra commentary.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "source_text": {"type": "string"},
        "target_text": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "source_text",
        "target_text",
        "confidence",
    ],
}

SINGLE_TEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
    },
    "required": ["text"],
}


@dataclass(frozen=True)
class SampleTriplet:
    sample_id: str
    original_video: Path
    edited_video: Path
    mask_video: Path


@dataclass(frozen=True)
class ProbeInfo:
    width: int
    height: int
    frame_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate bilingual text edit instructions from original/edited/mask videos via Ollama Qwen3-VL."
    )
    parser.add_argument("--original-dir", default="data/original_videos")
    parser.add_argument("--edited-dir", default="data/edited_videos")
    parser.add_argument("--mask-dir", default="data/original_text_masks")
    parser.add_argument("--output-dir", default="data/edit_instructions_qwen3vl")
    parser.add_argument("--model", default="qwen3-vl:32b-instruct")
    parser.add_argument("--fallback-model", default="qwen3-vl:8b-instruct")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--preflight-timeout", type=int, default=45)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--ids",
        nargs="*",
        default=[],
        help="Optional sample ids such as 0000007_00000 0000066_00000",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def probe_video(video_path: Path) -> ProbeInfo:
    output = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(video_path),
        ]
    )
    payload = json.loads(output)
    stream = payload["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    frame_count = 0
    nb_frames = stream.get("nb_frames")
    if isinstance(nb_frames, str) and nb_frames.isdigit():
        frame_count = int(nb_frames)
    if frame_count <= 0:
        duration = float(stream.get("duration") or 0.0)
        rate = stream.get("avg_frame_rate", "0/1")
        numerator, denominator = rate.split("/")
        fps = float(numerator) / float(denominator) if float(denominator) else 0.0
        frame_count = max(1, round(duration * fps))
    return ProbeInfo(width=width, height=height, frame_count=max(1, frame_count))


def frame_indices(frame_count: int) -> list[int]:
    fractions = (0.15, 0.50, 0.85)
    indices = sorted({min(frame_count - 1, max(0, round((frame_count - 1) * value))) for value in fractions})
    while len(indices) < 3:
        indices.append(indices[-1])
    return indices[:3]


def extract_frames(video_path: Path, output_dir: Path, prefix: str, indices: Iterable[int]) -> list[Path]:
    select_expr = "+".join(f"eq(n\\,{index})" for index in indices)
    output_pattern = output_dir / f"{prefix}_%02d.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"select='{select_expr}'",
            "-vsync",
            "0",
            str(output_pattern),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    frames = sorted(output_dir.glob(f"{prefix}_*.png"))
    if len(frames) != 3:
        raise RuntimeError(f"Expected 3 extracted frames for {video_path}, got {len(frames)}")
    return frames


def image_to_binary_mask(image_path: Path, threshold: int = 16) -> Image.Image:
    return Image.open(image_path).convert("L").point(lambda value: 255 if value > threshold else 0)


def raw_union_bbox(mask_frames: list[Path]) -> tuple[int, int, int, int]:
    boxes = []
    for mask_frame in mask_frames:
        bbox = image_to_binary_mask(mask_frame).getbbox()
        if bbox is not None:
            boxes.append(bbox)
    if not boxes:
        raise RuntimeError("Mask frames did not contain any visible region")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    scale_x: float,
    scale_y: float,
    min_pad_x: int,
    min_pad_y: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    pad_x = max(min_pad_x, int(width * scale_x))
    pad_y = max(min_pad_y, int(height * scale_y))
    image_width, image_height = image_size
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(image_width, x1 + pad_x),
        min(image_height, y1 + pad_y),
    )


def render_context_sheet(
    original_frame: Path,
    edited_frame: Path,
    mask_frame: Path,
    bbox: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    panels = []
    for label, image_path in (
        ("Original context", original_frame),
        ("Edited context", edited_frame),
        ("Mask context", mask_frame),
    ):
        image = Image.open(image_path).convert("RGB")
        drawer = ImageDraw.Draw(image)
        drawer.rectangle(bbox, outline=(255, 0, 0), width=6)
        image = image.resize((480, 270))
        canvas = Image.new("RGB", (480, 300), "white")
        canvas.paste(image, (0, 30))
        ImageDraw.Draw(canvas).text((10, 6), label, fill="black")
        panels.append(canvas)
    sheet = Image.new("RGB", (1440, 300), (240, 240, 240))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * 480, 0))
    sheet.save(output_path)


def render_crop_sheet(
    original_frames: list[Path],
    edited_frames: list[Path],
    bbox: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    panels = []
    labels = (
        "Original crop t1",
        "Original crop t2",
        "Original crop t3",
        "Edited crop t1",
        "Edited crop t2",
        "Edited crop t3",
    )
    for label, image_path in zip(labels, original_frames + edited_frames):
        image = Image.open(image_path).convert("RGB").crop(bbox)
        image = ImageOps.contain(image, (420, 220)).filter(ImageFilter.SHARPEN)
        canvas = Image.new("RGB", (440, 250), "white")
        offset_x = (440 - image.width) // 2
        offset_y = 30 + (220 - image.height) // 2
        canvas.paste(image, (offset_x, offset_y))
        ImageDraw.Draw(canvas).text((10, 6), label, fill="black")
        panels.append(canvas)
    sheet = Image.new("RGB", (1320, 500), (240, 240, 240))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 3) * 440, (index // 3) * 250))
    sheet.save(output_path)


def render_focus_sheet(
    original_frames: list[Path],
    edited_frames: list[Path],
    mask_frames: list[Path],
    bbox: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    panels = []
    labels = (
        "Original focus t1",
        "Original focus t2",
        "Original focus t3",
        "Edited focus t1",
        "Edited focus t2",
        "Edited focus t3",
    )
    for label, image_path, mask_path in zip(labels, original_frames + edited_frames, mask_frames + mask_frames):
        image = Image.open(image_path).convert("RGB").crop(bbox)
        mask = image_to_binary_mask(mask_path).crop(bbox)
        background = Image.new("RGB", image.size, "white")
        focused = Image.composite(image, background, mask)
        focused = ImageOps.contain(focused, (420, 220)).filter(ImageFilter.SHARPEN)
        canvas = Image.new("RGB", (440, 250), "white")
        offset_x = (440 - focused.width) // 2
        offset_y = 30 + (220 - focused.height) // 2
        canvas.paste(focused, (offset_x, offset_y))
        ImageDraw.Draw(canvas).text((10, 6), label, fill="black")
        panels.append(canvas)
    sheet = Image.new("RGB", (1320, 500), (240, 240, 240))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 3) * 440, (index // 3) * 250))
    sheet.save(output_path)


def render_single_focus_image(
    frame_path: Path,
    mask_path: Path,
    bbox: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    image = Image.open(frame_path).convert("RGB").crop(bbox)
    mask = image_to_binary_mask(mask_path).crop(bbox)
    background = Image.new("RGB", image.size, "white")
    focused = Image.composite(image, background, mask)
    focused = ImageOps.contain(focused, (900, 300)).filter(ImageFilter.SHARPEN)
    focused.save(output_path)


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def parse_model_json(raw_text: str) -> dict[str, object]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model did not return JSON: {raw_text}")
    return json.loads(cleaned[start : end + 1])


def validate_result(result: dict[str, object]) -> None:
    for key in ("source_text", "target_text", "confidence"):
        if key not in result:
            raise ValueError(f"Missing key: {key}")
    source_text = str(result["source_text"]).strip()
    target_text = str(result["target_text"]).strip()
    if not source_text or not target_text:
        raise ValueError("source_text or target_text is empty")


def build_bilingual_instructions(source_text: str, target_text: str) -> tuple[str, str]:
    return f"将{source_text}编辑为{target_text}", f"Change {source_text} to {target_text}"


def similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left.lower(), right.lower()).ratio()


def ping_model(url: str, model: str, timeout_seconds: int, keep_alive: str) -> bool:
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": "Reply with OK only."}],
        "options": {"temperature": 0, "num_predict": 4},
        "keep_alive": keep_alive,
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
    except Exception:
        return False
    return "OK" in content


def select_model(args: argparse.Namespace) -> tuple[str, bool]:
    if ping_model(args.ollama_url, args.model, args.preflight_timeout, args.keep_alive):
        return args.model, False
    if args.fallback_model and ping_model(
        args.ollama_url, args.fallback_model, args.preflight_timeout, args.keep_alive
    ):
        return args.fallback_model, True
    raise RuntimeError(
        f"Neither {args.model!r} nor fallback model {args.fallback_model!r} responded successfully."
    )


def request_instruction(
    url: str,
    model: str,
    keep_alive: str,
    timeout_seconds: int,
    context_image: Path,
    crop_image: Path,
    focus_image: Path,
) -> dict[str, object]:
    payload = {
        "model": model,
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "options": {"temperature": 0, "num_predict": 160},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT,
                "images": [
                    encode_image(context_image),
                    encode_image(crop_image),
                    encode_image(focus_image),
                ],
            },
        ],
        "keep_alive": keep_alive,
    }
    response = requests.post(url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    message = response.json()["message"]["content"]
    result = parse_model_json(message)
    validate_result(result)
    return result


def request_single_text(
    url: str,
    model: str,
    keep_alive: str,
    timeout_seconds: int,
    image_path: Path,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "format": SINGLE_TEXT_SCHEMA,
        "options": {"temperature": 0, "num_predict": 120},
        "messages": [
            {
                "role": "user",
                "content": (
                    "Read the exact visible text in this single masked-focus image. "
                    "Return JSON with one key named text. Join multiple lines with spaces. "
                    "Return JSON only."
                ),
                "images": [encode_image(image_path)],
            }
        ],
        "keep_alive": keep_alive,
    }
    response = requests.post(url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    message = response.json()["message"]["content"]
    payload = parse_model_json(message)
    return str(payload.get("text", "")).strip()


def refine_same_text_result(
    baseline_text: str,
    original_ocr: str,
    edited_ocr: str,
) -> tuple[str, str]:
    if not original_ocr or not edited_ocr or original_ocr == edited_ocr:
        return baseline_text, baseline_text

    refined_source = original_ocr
    refined_target = edited_ocr

    if baseline_text != edited_ocr and similarity(baseline_text, edited_ocr) >= 0.65 and len(baseline_text) > len(edited_ocr):
        refined_target = baseline_text
    if baseline_text != original_ocr and similarity(baseline_text, original_ocr) >= 0.65 and len(baseline_text) > len(original_ocr):
        refined_source = baseline_text

    if refined_source == refined_target:
        return baseline_text, baseline_text
    return refined_source, refined_target


def build_triplets(original_dir: Path, edited_dir: Path, mask_dir: Path) -> list[SampleTriplet]:
    original_map = {path.stem: path for path in sorted(original_dir.glob("*.mp4"))}
    edited_map: dict[str, Path] = {}
    for path in sorted(edited_dir.glob("*.mp4")):
        sample_id = path.stem.removesuffix("_overlay")
        edited_map[sample_id] = path
    mask_map: dict[str, Path] = {}
    for path in sorted(mask_dir.glob("*.mp4")):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        sample_id = "_".join(parts[:2])
        mask_map[sample_id] = path
    sample_ids = sorted(set(original_map) & set(edited_map) & set(mask_map))
    return [
        SampleTriplet(
            sample_id=sample_id,
            original_video=original_map[sample_id],
            edited_video=edited_map[sample_id],
            mask_video=mask_map[sample_id],
        )
        for sample_id in sample_ids
    ]


def maybe_filter_triplets(triplets: list[SampleTriplet], ids: list[str], limit: int) -> list[SampleTriplet]:
    if ids:
        allowed = set(ids)
        triplets = [triplet for triplet in triplets if triplet.sample_id in allowed]
    if limit > 0:
        triplets = triplets[:limit]
    return triplets


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def save_failure(output_dir: Path, sample_id: str, error_message: str) -> None:
    payload: dict[str, object] = {
        "sample_id": sample_id,
        "error": error_message,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / f"{sample_id}.error.json", payload)


def rebuild_index(output_dir: Path) -> None:
    records = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name in {"manifest.json", "all_instructions.json"} or path.name.endswith(".error.json"):
            continue
        if path.stem.count("_") != 1:
            continue
        records.append(json.loads(path.read_text()))
    write_json(output_dir / "all_instructions.json", {"count": len(records), "records": records})


def process_sample(
    triplet: SampleTriplet,
    output_dir: Path,
    model_requested: str,
    model_used: str,
    args: argparse.Namespace,
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"{triplet.sample_id}_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        original_probe = probe_video(triplet.original_video)
        edited_probe = probe_video(triplet.edited_video)
        mask_probe = probe_video(triplet.mask_video)

        original_frames = extract_frames(
            triplet.original_video, temp_dir, "original", frame_indices(original_probe.frame_count)
        )
        edited_frames = extract_frames(
            triplet.edited_video, temp_dir, "edited", frame_indices(edited_probe.frame_count)
        )
        mask_frames = extract_frames(triplet.mask_video, temp_dir, "mask", frame_indices(mask_probe.frame_count))
        raw_bbox = raw_union_bbox(mask_frames)
        bbox = expand_bbox(raw_bbox, (original_probe.width, original_probe.height), 0.25, 0.40, 20, 12)
        focus_bbox = expand_bbox(raw_bbox, (original_probe.width, original_probe.height), 0.05, 0.10, 4, 4)

        context_sheet = temp_dir / "context_sheet.png"
        crop_sheet = temp_dir / "crop_sheet.png"
        focus_sheet = temp_dir / "focus_sheet.png"
        original_focus_single = temp_dir / "original_focus_single.png"
        edited_focus_single = temp_dir / "edited_focus_single.png"
        render_context_sheet(original_frames[1], edited_frames[1], mask_frames[1], bbox, context_sheet)
        render_crop_sheet(original_frames, edited_frames, bbox, crop_sheet)
        render_focus_sheet(original_frames, edited_frames, mask_frames, focus_bbox, focus_sheet)
        render_single_focus_image(original_frames[1], mask_frames[1], focus_bbox, original_focus_single)
        render_single_focus_image(edited_frames[1], mask_frames[1], focus_bbox, edited_focus_single)

        result = request_instruction(
            url=args.ollama_url,
            model=model_used,
            keep_alive=args.keep_alive,
            timeout_seconds=args.request_timeout,
            context_image=context_sheet,
            crop_image=crop_sheet,
            focus_image=focus_sheet,
        )
        source_text = str(result["source_text"]).strip()
        target_text = str(result["target_text"]).strip()
        if source_text == target_text:
            original_ocr = request_single_text(
                url=args.ollama_url,
                model=model_used,
                keep_alive=args.keep_alive,
                timeout_seconds=args.request_timeout,
                image_path=original_focus_single,
            )
            edited_ocr = request_single_text(
                url=args.ollama_url,
                model=model_used,
                keep_alive=args.keep_alive,
                timeout_seconds=args.request_timeout,
                image_path=edited_focus_single,
            )
            source_text, target_text = refine_same_text_result(source_text, original_ocr, edited_ocr)
        instruction_zh, instruction_en = build_bilingual_instructions(source_text, target_text)

        payload: dict[str, object] = {
            "sample_id": triplet.sample_id,
            "original_video": str(triplet.original_video),
            "edited_video": str(triplet.edited_video),
            "mask_video": str(triplet.mask_video),
            "source_text": source_text,
            "target_text": target_text,
            "instruction_zh": instruction_zh,
            "instruction_en": instruction_en,
            "confidence": result["confidence"],
            "model_requested": model_requested,
            "model_used": model_used,
            "bbox_xyxy": list(bbox),
            "focus_bbox_xyxy": list(focus_bbox),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output_dir / f"{triplet.sample_id}.json", payload)


def main() -> None:
    args = parse_args()
    original_dir = Path(args.original_dir)
    edited_dir = Path(args.edited_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    triplets = build_triplets(original_dir, edited_dir, mask_dir)
    triplets = maybe_filter_triplets(triplets, args.ids, args.limit)
    model_used, used_fallback = select_model(args)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "original_dir": str(original_dir),
        "edited_dir": str(edited_dir),
        "mask_dir": str(mask_dir),
        "output_dir": str(output_dir),
        "requested_model": args.model,
        "fallback_model": args.fallback_model,
        "active_model": model_used,
        "used_fallback": used_fallback,
        "sample_count": len(triplets),
    }
    write_json(output_dir / "manifest.json", manifest)

    total = len(triplets)
    for index, triplet in enumerate(triplets, start=1):
        output_path = output_dir / f"{triplet.sample_id}.json"
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{total}] skip {triplet.sample_id}")
            continue
        started_at = time.time()
        try:
            process_sample(triplet, output_dir, args.model, model_used, args)
            error_path = output_dir / f"{triplet.sample_id}.error.json"
            if error_path.exists():
                error_path.unlink()
            elapsed = time.time() - started_at
            print(f"[{index}/{total}] ok   {triplet.sample_id} ({elapsed:.2f}s)")
        except Exception as error:  # noqa: BLE001
            save_failure(output_dir, triplet.sample_id, str(error))
            print(f"[{index}/{total}] fail {triplet.sample_id}: {error}")

    rebuild_index(output_dir)


if __name__ == "__main__":
    main()
