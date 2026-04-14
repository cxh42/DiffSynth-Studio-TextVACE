"""
Metric 5: Ground Truth Similarity (for paired samples only)
=============================================================
Measures how close the edited video is to the ground-truth edited video.

This is the GOLD STANDARD metric — only possible because we have
high-quality real GT video pairs. Synthetic benchmarks cannot provide this.

Computes:
  - gt_psnr: PSNR between edited and GT (full frame)
  - gt_ssim: SSIM between edited and GT (full frame)
  - gt_text_psnr: PSNR in mask region only (text rendering quality)
  - gt_text_ssim: SSIM in mask region only
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def evaluate_gt_similarity(
    edited_video_path: str,
    gt_video_path: str,
    mask_video_path: str,
) -> dict:
    """Evaluate similarity between edited video and ground truth.

    Args:
        edited_video_path: path to the method's edited video
        gt_video_path: path to the ground-truth edited video
        mask_video_path: path to the mask video

    Returns:
        dict with gt_psnr, gt_ssim, gt_text_psnr, gt_text_ssim
    """
    cap_e = cv2.VideoCapture(edited_video_path)
    cap_g = cv2.VideoCapture(gt_video_path)
    cap_m = cv2.VideoCapture(mask_video_path)

    full_psnr = []
    full_ssim = []
    text_psnr = []
    text_ssim = []

    while True:
        ret_e, frame_e = cap_e.read()
        ret_g, frame_g = cap_g.read()
        ret_m, frame_m = cap_m.read()
        if not ret_e or not ret_g or not ret_m:
            break

        # Ensure same size
        h, w = frame_g.shape[:2]
        if frame_e.shape[:2] != (h, w):
            frame_e = cv2.resize(frame_e, (w, h))
        if frame_m.shape[:2] != (h, w):
            frame_m = cv2.resize(frame_m, (w, h))

        # Full frame PSNR/SSIM
        mse_full = np.mean((frame_e.astype(float) - frame_g.astype(float)) ** 2)
        if mse_full < 1e-10:
            full_psnr.append(100.0)
        else:
            full_psnr.append(10 * np.log10(255.0 ** 2 / mse_full))

        ssim_full = structural_similarity(frame_e, frame_g, channel_axis=2, data_range=255)
        full_ssim.append(ssim_full)

        # Text region (mask) PSNR/SSIM
        mask_gray = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        mask_float = mask_bin.astype(np.float32) / 255.0

        n_text_pixels = mask_float.sum() * 3  # 3 channels
        if n_text_pixels < 100:
            continue

        # PSNR on text region
        diff_sq = (frame_e.astype(float) - frame_g.astype(float)) ** 2
        mse_text = np.sum(diff_sq * mask_float[:, :, np.newaxis]) / n_text_pixels
        if mse_text < 1e-10:
            text_psnr.append(100.0)
        else:
            text_psnr.append(10 * np.log10(255.0 ** 2 / mse_text))

        # SSIM on text region bounding box
        ys, xs = np.where(mask_bin > 0)
        if len(ys) > 0:
            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1
            if (y2 - y1) >= 7 and (x2 - x1) >= 7:
                ssim_text = structural_similarity(
                    frame_e[y1:y2, x1:x2], frame_g[y1:y2, x1:x2],
                    channel_axis=2, data_range=255,
                )
                text_ssim.append(ssim_text)

    cap_e.release()
    cap_g.release()
    cap_m.release()

    if not full_psnr:
        return {
            "gt_psnr": 0.0, "gt_ssim": 0.0,
            "gt_text_psnr": 0.0, "gt_text_ssim": 0.0,
            "frames_evaluated": 0,
        }

    return {
        "gt_psnr": float(np.mean(full_psnr)),
        "gt_ssim": float(np.mean(full_ssim)),
        "gt_text_psnr": float(np.mean(text_psnr)) if text_psnr else 0.0,
        "gt_text_ssim": float(np.mean(text_ssim)) if text_ssim else 0.0,
        "frames_evaluated": len(full_psnr),
    }
