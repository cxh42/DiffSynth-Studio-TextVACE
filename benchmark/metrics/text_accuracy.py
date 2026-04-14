"""
Metric 1: Text Accuracy
========================
Measures whether the target text was correctly rendered in the edited video.

Uses EasyOCR to detect and recognize text in the FULL frame, then finds
detections that fall inside the mask region and compares with target text.

Key improvement over crop-based approach: full-frame OCR avoids issues with
small/misaligned crops where OCR detection fails entirely.

Outputs:
  - word_accuracy: fraction of frames where OCR in mask region exactly matches target
  - char_accuracy: 1 - normalized edit distance (character-level)
  - ocr_confidence: mean OCR confidence for mask-region detections
  - detection_rate: fraction of frames where text is detected in mask region
"""

import cv2
import numpy as np
import easyocr


def _normalized_edit_distance(pred: str, target: str) -> float:
    """Compute normalized edit distance between two strings."""
    m, n = len(pred), len(target)
    if m == 0 and n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / max(m, n, 1)


def _find_text_in_mask(detections, mask_binary):
    """Find EasyOCR detections whose center falls inside the mask region.

    Args:
        detections: EasyOCR results [(box, text, conf), ...]
        mask_binary: binary mask (>127 = text region)

    Returns:
        list of (text, confidence)
    """
    matches = []
    for (box, text, conf) in detections:
        pts = np.array(box)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())

        if 0 <= cy < mask_binary.shape[0] and 0 <= cx < mask_binary.shape[1]:
            if mask_binary[cy, cx] > 127:
                matches.append((text, conf))

    return matches


def evaluate_text_accuracy(
    edited_video_path: str,
    mask_video_path: str,
    target_text: str,
    ocr_engine=None,
    sample_interval: int = 6,
) -> dict:
    """Evaluate text accuracy using full-frame OCR + mask filtering.

    Args:
        edited_video_path: path to the edited video
        mask_video_path: path to the mask video (white = text region)
        target_text: ground-truth target text string
        ocr_engine: EasyOCR reader instance (reuse across samples)
        sample_interval: evaluate every N frames

    Returns:
        dict with word_accuracy, char_accuracy, ocr_confidence, detection_rate
    """
    if ocr_engine is None:
        ocr_engine = easyocr.Reader(["en", "ch_sim"], gpu=True, verbose=False)

    cap_e = cv2.VideoCapture(edited_video_path)
    cap_m = cv2.VideoCapture(mask_video_path)

    target_clean = target_text.strip().lower()

    word_matches = []
    char_accuracies = []
    confidences = []
    detected = []

    fi = 0
    while True:
        ret_e, frame_e = cap_e.read()
        ret_m, frame_m = cap_m.read()
        if not ret_e or not ret_m:
            break

        if fi % sample_interval != 0:
            fi += 1
            continue

        if frame_m.shape[:2] != frame_e.shape[:2]:
            frame_m = cv2.resize(frame_m, (frame_e.shape[1], frame_e.shape[0]))

        frame_rgb = cv2.cvtColor(frame_e, cv2.COLOR_BGR2RGB)
        mask_gray = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)

        # Full-frame OCR detection + recognition
        all_detections = ocr_engine.readtext(frame_rgb)

        # Filter: only detections whose center is inside mask
        mask_texts = _find_text_in_mask(all_detections, mask_gray)

        if len(mask_texts) == 0:
            detected.append(False)
            fi += 1
            continue

        detected.append(True)

        # Concatenate all detected text in mask region
        all_text = " ".join([t for t, c in mask_texts]).strip().lower()
        avg_conf = np.mean([c for t, c in mask_texts])
        confidences.append(avg_conf)

        # Word accuracy: exact match
        word_matches.append(all_text == target_clean)

        # Character accuracy: 1 - NED
        ned = _normalized_edit_distance(all_text, target_clean)
        char_accuracies.append(1.0 - ned)

        fi += 1

    cap_e.release()
    cap_m.release()

    n_evaluated = len(detected)
    if n_evaluated == 0:
        return {
            "word_accuracy": 0.0,
            "char_accuracy": 0.0,
            "ocr_confidence": 0.0,
            "detection_rate": 0.0,
            "frames_evaluated": 0,
        }

    return {
        "word_accuracy": float(np.mean(word_matches)) if word_matches else 0.0,
        "char_accuracy": float(np.mean(char_accuracies)) if char_accuracies else 0.0,
        "ocr_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "detection_rate": float(np.mean(detected)),
        "frames_evaluated": n_evaluated,
    }
