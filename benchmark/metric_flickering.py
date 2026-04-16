"""
Metric: Flickering (Temporal Stability, VBench style)
======================================================
全帧相邻帧像素级 MAE → VBench 归一化 (255 - MAE) / 255。
越高越稳定。纯像素运算，无需模型。
"""

import cv2
import numpy as np


def compute(edited_frames):
    """
    Args:
        edited_frames: list of RGB numpy arrays (H, W, 3)

    Returns:
        float in [0, 1], higher = more stable (less flickering)
    """
    n = len(edited_frames)
    if n < 2:
        return 1.0

    maes = []
    for i in range(n - 1):
        diff = cv2.absdiff(edited_frames[i], edited_frames[i + 1])
        maes.append(np.mean(diff))

    mean_mae = float(np.mean(maes))
    return (255.0 - mean_mae) / 255.0
