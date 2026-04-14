"""
Metric 3: Temporal Consistency
===============================
Measures whether the edited text region is stable across frames
(no flickering, jittering, or sudden changes).

Two sub-metrics:
  - text_temporal_ssim: SSIM between consecutive frames in the mask region
  - clip_frame_consistency: CLIP feature cosine similarity between consecutive frames

Inspired by VBench's Temporal Flickering and Subject Consistency dimensions.
"""

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity


def _crop_mask_bbox(frame, mask, padding=20):
    """Get bounding box crop of mask region with generous padding."""
    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        return None
    y1 = max(0, ys.min() - padding)
    y2 = min(frame.shape[0], ys.max() + padding + 1)
    x1 = max(0, xs.min() - padding)
    x2 = min(frame.shape[1], xs.max() + padding + 1)
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    return crop


def evaluate_temporal_consistency(
    edited_video_path: str,
    mask_video_path: str,
    use_clip: bool = True,
    clip_model=None,
    clip_processor=None,
    device: str = "cuda",
) -> dict:
    """Evaluate temporal consistency of edited video in the text region.

    Args:
        edited_video_path: path to the edited video
        mask_video_path: path to the mask video
        use_clip: whether to compute CLIP frame consistency
        clip_model: preloaded CLIP model (reuse across samples)
        clip_processor: preloaded CLIP processor
        device: torch device

    Returns:
        dict with text_temporal_ssim, clip_frame_consistency
    """
    cap_e = cv2.VideoCapture(edited_video_path)
    cap_m = cv2.VideoCapture(mask_video_path)

    # Load CLIP if needed and not provided
    if use_clip and clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    prev_crop = None
    prev_clip_feat = None
    ssim_values = []
    clip_sims = []

    while True:
        ret_e, frame_e = cap_e.read()
        ret_m, frame_m = cap_m.read()
        if not ret_e or not ret_m:
            break

        mask_gray = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
        crop = _crop_mask_bbox(frame_e, mask_gray, padding=5)
        if crop is None or crop.shape[0] < 5 or crop.shape[1] < 5:
            prev_crop = None
            prev_clip_feat = None
            continue

        # SSIM between consecutive frames in mask region
        if prev_crop is not None:
            # Resize to same shape for comparison
            h = min(prev_crop.shape[0], crop.shape[0])
            w = min(prev_crop.shape[1], crop.shape[1])
            c1 = cv2.resize(prev_crop, (w, h))
            c2 = cv2.resize(crop, (w, h))
            if h >= 7 and w >= 7:
                ssim_val = structural_similarity(c1, c2, channel_axis=2, data_range=255)
                ssim_values.append(ssim_val)

        prev_crop = crop

        # CLIP frame consistency
        if use_clip and clip_model is not None:
            from PIL import Image
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            inputs = clip_processor(images=pil_img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                out = clip_model.get_image_features(pixel_values=pixel_values)
                feat = out.pooler_output if hasattr(out, 'pooler_output') else out
                feat = feat / feat.norm(dim=-1, keepdim=True)

            if prev_clip_feat is not None:
                sim = torch.cosine_similarity(prev_clip_feat, feat).item()
                clip_sims.append(sim)

            prev_clip_feat = feat

    cap_e.release()
    cap_m.release()

    result = {
        "text_temporal_ssim": float(np.mean(ssim_values)) if ssim_values else 0.0,
        "frames_evaluated": len(ssim_values),
    }

    if use_clip:
        result["clip_frame_consistency"] = float(np.mean(clip_sims)) if clip_sims else 0.0

    return result
