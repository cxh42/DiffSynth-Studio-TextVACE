"""
Metric: MUSIQ (Imaging Quality, VBench style)
===============================================
对每帧 resize 到 max 512px，输入 MUSIQ（KonIQ 预训练）得到质量分，
取所有帧均值。衡量整体图像感知质量。
"""

import cv2
import numpy as np
import torch


_MUSIQ_MODEL = None


def get_musiq_model(device="cuda"):
    """Lazy-load MUSIQ model (cached)."""
    global _MUSIQ_MODEL
    if _MUSIQ_MODEL is None:
        import pyiqa
        _MUSIQ_MODEL = pyiqa.create_metric("musiq", device=device)
    return _MUSIQ_MODEL


def compute(edited_frames, device="cuda"):
    """
    Args:
        edited_frames: list of RGB numpy arrays (H, W, 3)
        device: cuda or cpu

    Returns:
        float in [0, 100], higher is better
    """
    model = get_musiq_model(device)
    scores = []

    for frame in edited_frames:
        # VBench-style: resize to max 512px before scoring
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        if max_dim > 512:
            scale = 512.0 / max_dim
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        frame_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        frame_t = frame_t.to(device)
        with torch.no_grad():
            score = model(frame_t)
        scores.append(float(score.item()))

    return float(np.mean(scores)) if scores else 0.0
