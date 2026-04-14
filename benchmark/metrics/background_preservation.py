"""
Metric 2: Background Preservation
===================================
Measures whether non-text regions remain unchanged after editing.

Compares the edited video against the original video, but ONLY in the
regions outside the text mask (mask_inv = 1 - mask).

Outputs:
  - bg_psnr: Peak Signal-to-Noise Ratio on non-mask region
  - bg_ssim: Structural Similarity on non-mask region
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def evaluate_background_preservation(
    original_video_path: str,
    edited_video_path: str,
    mask_video_path: str,
) -> dict:
    """Evaluate background preservation between original and edited video.

    Args:
        original_video_path: path to the original video (before editing)
        edited_video_path: path to the edited video (after editing)
        mask_video_path: path to the mask video (white = text region)

    Returns:
        dict with bg_psnr, bg_ssim, frames_evaluated
    """
    cap_o = cv2.VideoCapture(original_video_path)
    cap_e = cv2.VideoCapture(edited_video_path)
    cap_m = cv2.VideoCapture(mask_video_path)

    psnr_values = []
    ssim_values = []
    fi = 0

    while True:
        ret_o, frame_o = cap_o.read()
        ret_e, frame_e = cap_e.read()
        ret_m, frame_m = cap_m.read()
        if not ret_o or not ret_e or not ret_m:
            break

        # Ensure same size
        if frame_o.shape != frame_e.shape:
            frame_e = cv2.resize(frame_e, (frame_o.shape[1], frame_o.shape[0]))
        if frame_m.shape[:2] != frame_o.shape[:2]:
            frame_m = cv2.resize(frame_m, (frame_o.shape[1], frame_o.shape[0]))

        # Create inverse mask (non-text region)
        mask_gray = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        mask_inv = (mask_bin == 0).astype(np.uint8)  # 1 = background

        # Check if there's enough background to evaluate
        bg_ratio = mask_inv.sum() / mask_inv.size
        if bg_ratio < 0.1:
            fi += 1
            continue

        # Apply mask: keep only background pixels
        orig_bg = frame_o * mask_inv[:, :, np.newaxis]
        edit_bg = frame_e * mask_inv[:, :, np.newaxis]

        # PSNR on background region
        # To avoid counting black masked pixels, compute on the valid region only
        ys, xs = np.where(mask_inv > 0)
        if len(ys) == 0:
            fi += 1
            continue

        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1

        # Compute PSNR on the full frame but masked
        # Use data_range=255 for uint8 images
        mse = np.mean((orig_bg.astype(float) - edit_bg.astype(float)) ** 2 * mask_inv[:, :, np.newaxis])
        # Adjust MSE for only background pixels
        n_bg_pixels = mask_inv.sum() * 3  # 3 channels
        mse_adjusted = np.sum((orig_bg.astype(float) - edit_bg.astype(float)) ** 2) / n_bg_pixels
        if mse_adjusted < 1e-10:
            psnr_val = 100.0  # essentially perfect
        else:
            psnr_val = 10 * np.log10(255.0 ** 2 / mse_adjusted)
        psnr_values.append(psnr_val)

        # SSIM on the bounding box of background (approximate)
        # Use the full frame with mask applied
        ssim_val = structural_similarity(
            orig_bg[y1:y2, x1:x2],
            edit_bg[y1:y2, x1:x2],
            channel_axis=2,
            data_range=255,
        )
        ssim_values.append(ssim_val)

        fi += 1

    cap_o.release()
    cap_e.release()
    cap_m.release()

    if len(psnr_values) == 0:
        return {"bg_psnr": 0.0, "bg_ssim": 0.0, "frames_evaluated": 0}

    return {
        "bg_psnr": float(np.mean(psnr_values)),
        "bg_ssim": float(np.mean(ssim_values)),
        "frames_evaluated": fi,
    }
