"""Shared utilities: video/mask loading and bbox extraction."""

import cv2
import numpy as np


def load_video_frames(video_path):
    """Load all frames from a video as RGB numpy arrays (H, W, 3)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_mask_frames(mask_path, target_h=None, target_w=None):
    """Load ALL mask frames as binary arrays (H, W), optionally resizing.

    Returns list of binary uint8 masks, one per frame.
    """
    cap = cv2.VideoCapture(mask_path)
    masks = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if target_h is not None and target_w is not None:
            if gray.shape[0] != target_h or gray.shape[1] != target_w:
                gray = cv2.resize(gray, (target_w, target_h),
                                  interpolation=cv2.INTER_NEAREST)
        _, mask_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        masks.append(mask_bin)
    cap.release()
    return masks


def get_mask_bbox(mask_bin, pad_ratio=0.1):
    """Get axis-aligned bounding box of mask region with padding."""
    ys, xs = np.where(mask_bin > 0)
    if len(ys) == 0:
        return None
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    h, w = mask_bin.shape[:2]
    pad_h = int((y2 - y1) * pad_ratio)
    pad_w = int((x2 - x1) * pad_ratio)
    y1 = max(0, y1 - pad_h)
    y2 = min(h, y2 + pad_h)
    x1 = max(0, x1 - pad_w)
    x2 = min(w, x2 + pad_w)

    return x1, y1, x2, y2
