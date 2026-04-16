"""
Metric: SeqAcc (Sequence Accuracy)
===================================
逐帧完全匹配率：OCR 识别结果与目标文字完全相同的帧数 / 总帧数。
最严格的指标——差一个字符就判为失败。
"""

import numpy as np


def compute(ocr_per_frame, target_text):
    """
    Args:
        ocr_per_frame: list of OCR text strings, one per frame
        target_text: ground truth target text string

    Returns:
        float in [0, 1], higher is better
    """
    target = target_text.strip().upper()
    if not ocr_per_frame:
        return 0.0

    matches = [
        1.0 if pred.strip().upper() == target else 0.0
        for pred in ocr_per_frame
    ]
    return float(np.mean(matches))
