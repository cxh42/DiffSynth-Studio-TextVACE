"""
Metric: SSIM (non-text region)
================================
只对非文字区域计算编辑视频与原始视频的 SSIM。
做法：将编辑视频的文字区域像素设为原始视频对应像素，让文字区域"抵消"不参与比较。
"""

import numpy as np
from skimage.metrics import structural_similarity


def compute_single_frame(orig_frame, edited_frame, mask_bin):
    """Compute SSIM with text region neutralized."""
    o = orig_frame.copy()
    e = edited_frame.copy()
    # Make text region identical in both → contributes perfect similarity and cancels
    mask_text = mask_bin > 0
    e[mask_text] = o[mask_text]
    return float(structural_similarity(o, e, channel_axis=2, data_range=255))


def compute(orig_frames, edited_frames, mask_frames):
    """
    Args:
        orig_frames: list of RGB arrays (original video)
        edited_frames: list of RGB arrays (edited video)
        mask_frames: list of binary mask arrays (per-frame)

    Returns:
        mean SSIM in [0, 1] over all frames, higher is better
    """
    scores = []
    for i in range(len(edited_frames)):
        mask_idx = min(i, len(mask_frames) - 1)
        scores.append(compute_single_frame(
            orig_frames[i], edited_frames[i], mask_frames[mask_idx]
        ))
    return float(np.mean(scores)) if scores else 0.0
