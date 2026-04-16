"""
Metric: FVD (Frechet Video Distance)
======================================
全局指标：对整个测试集的编辑视频和 GT 参考视频提取 R3D-18（Kinetics-400）
视频级特征，拟合高斯分布后计算 Frechet 距离。

参考分布：数据集自带的高质量编辑后视频（data/raw/edited_videos/）。
整个测试集算一个值，不是逐样本。
"""

import os
import sys

import cv2
import numpy as np
import torch
from scipy.linalg import sqrtm


_R3D_MODEL = None


def get_r3d_model(device="cuda"):
    """Lazy-load R3D-18 (Kinetics-400 pretrained, fc→Identity)."""
    global _R3D_MODEL
    if _R3D_MODEL is None:
        from torchvision.models.video import r3d_18, R3D_18_Weights
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1).to(device)
        model.eval()
        model.fc = torch.nn.Identity()
        _R3D_MODEL = model
    return _R3D_MODEL


def extract_video_features(video_frames, device="cuda", max_frames=16):
    """Extract 512-dim R3D-18 features from a single video."""
    model = get_r3d_model(device)

    n = len(video_frames)
    indices = np.linspace(0, n - 1, max_frames, dtype=int)
    frames = [video_frames[i] for i in indices]

    # Resize to 112x112 (R3D-18 expected input)
    resized = [cv2.resize(f, (112, 112)) for f in frames]
    tensor = np.stack(resized)  # (T, 112, 112, 3)
    tensor = torch.from_numpy(tensor).permute(3, 0, 1, 2).unsqueeze(0).float().to(device) / 255.0

    # Kinetics-400 normalization
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1).to(device)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1).to(device)
    tensor = (tensor - mean) / std

    with torch.no_grad():
        feat = model(tensor)  # (1, 512)
    return feat[0].cpu().numpy()


def _frechet_distance(feats_real, feats_fake):
    """Compute Frechet distance between two Gaussian distributions."""
    mu_real = np.mean(feats_real, axis=0)
    mu_fake = np.mean(feats_fake, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    sigma_fake = np.cov(feats_fake, rowvar=False)

    # Regularize for numerical stability when N < feature_dim
    eps = 1e-6
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_fake += eps * np.eye(sigma_fake.shape[0])

    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fvd)


def compute(edited_videos, ref_videos, device="cuda"):
    """
    Args:
        edited_videos: list of [list of RGB frames] — the method's outputs
        ref_videos: list of [list of RGB frames] — high-quality reference videos
        device: cuda or cpu

    Returns:
        float ≥ 0, lower is better
    """
    print(f"  Extracting features from {len(edited_videos)} edited videos...")
    feats_edited = [extract_video_features(v, device) for v in edited_videos]

    print(f"  Extracting features from {len(ref_videos)} reference videos...")
    feats_ref = [extract_video_features(v, device) for v in ref_videos]

    feats_edited = np.stack(feats_edited)
    feats_ref = np.stack(feats_ref)

    return _frechet_distance(feats_ref, feats_edited)
