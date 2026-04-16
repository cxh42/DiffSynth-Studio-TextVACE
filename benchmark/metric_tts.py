"""
Metric: TTS (Temporal Text Stability)
======================================
相邻帧 OCR 结果完全相同的帧对数 / 总帧对数（T-1 对）。
不关心文字对错，只关心帧间一致性。专门暴露逐帧图像编辑方法的抖动弱点。
"""

import numpy as np


def compute(ocr_per_frame):
    """
    Args:
        ocr_per_frame: list of OCR text strings, one per frame

    Returns:
        float in [0, 1], higher is better (more stable)
    """
    n = len(ocr_per_frame)
    if n < 2:
        return 1.0

    norm = [t.strip().upper() for t in ocr_per_frame]
    matches = [1.0 if norm[i] == norm[i + 1] else 0.0 for i in range(n - 1)]
    return float(np.mean(matches))
