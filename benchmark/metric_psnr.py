"""
Metric: PSNR (non-text region)
================================
只对非文字区域（掩码取反）计算编辑视频与原始视频的 PSNR。
完美保留背景时 PSNR → ∞（我们 cap 在 100 dB），整张图被重建则显著下降。
"""

import numpy as np


def compute_single_frame(orig_frame, edited_frame, mask_bin):
    """Compute PSNR on non-text region (mask_bin == 0 means non-text)."""
    mask_inv = mask_bin == 0
    pixels_orig = orig_frame[mask_inv].astype(np.float64)
    pixels_edit = edited_frame[mask_inv].astype(np.float64)
    if pixels_orig.size == 0:
        return 100.0
    mse = np.mean((pixels_orig - pixels_edit) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def compute(orig_frames, edited_frames, mask_frames):
    """
    Args:
        orig_frames: list of RGB arrays (original video)
        edited_frames: list of RGB arrays (edited video)
        mask_frames: list of binary mask arrays (per-frame)

    Returns:
        mean PSNR (dB) over all frames, higher is better
    """
    scores = []
    for i in range(len(edited_frames)):
        mask_idx = min(i, len(mask_frames) - 1)
        scores.append(compute_single_frame(
            orig_frames[i], edited_frames[i], mask_frames[mask_idx]
        ))
    return float(np.mean(scores)) if scores else 0.0
