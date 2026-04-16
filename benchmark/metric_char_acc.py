"""
Metric: CharAcc (Character Accuracy)
=====================================
逐帧字符级准确率，基于归一化 Levenshtein 编辑距离：
    score = 1 - edit_distance(pred, target) / max(len(pred), len(target))
取 120 帧的均值。允许部分字符正确。
"""

import numpy as np


def levenshtein_distance(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(curr_row[j] + 1,
                                prev_row[j + 1] + 1,
                                prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[-1]


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

    accs = []
    for pred in ocr_per_frame:
        pred = pred.strip().upper()
        if len(pred) == 0 and len(target) == 0:
            accs.append(1.0)
        elif len(pred) == 0 or len(target) == 0:
            accs.append(0.0)
        else:
            ed = levenshtein_distance(pred, target)
            accs.append(1.0 - ed / max(len(pred), len(target)))
    return float(np.mean(accs))
