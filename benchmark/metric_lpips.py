"""
Metric: LPIPS (non-text region)
=================================
将编辑视频的文字区域像素设为原始视频对应像素，让文字区域不贡献距离，
只测量背景的感知差异。LPIPS 使用 AlexNet backbone（CVPR 2018）。
"""

import numpy as np
import torch


_LPIPS_MODEL = None


def get_lpips_model(device="cuda"):
    """Lazy-load LPIPS(alex) model (cached)."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device)
    return _LPIPS_MODEL


def compute_single_frame(orig_frame, edited_frame, mask_bin, device="cuda"):
    """Compute LPIPS with text region set to original pixel values."""
    o = orig_frame.copy()
    e = edited_frame.copy()
    mask_text = mask_bin > 0
    e[mask_text] = o[mask_text]

    model = get_lpips_model(device)
    o_t = torch.from_numpy(o).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0
    e_t = torch.from_numpy(e).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0

    with torch.no_grad():
        d = model(o_t, e_t)
    return float(d.item())


def compute(orig_frames, edited_frames, mask_frames, device="cuda"):
    """
    Args:
        orig_frames: list of RGB arrays
        edited_frames: list of RGB arrays
        mask_frames: list of binary mask arrays (per-frame)

    Returns:
        mean LPIPS in [0, 1] over all frames, lower is better
    """
    scores = []
    for i in range(len(edited_frames)):
        mask_idx = min(i, len(mask_frames) - 1)
        scores.append(compute_single_frame(
            orig_frames[i], edited_frames[i], mask_frames[mask_idx], device
        ))
    return float(np.mean(scores)) if scores else 0.0
