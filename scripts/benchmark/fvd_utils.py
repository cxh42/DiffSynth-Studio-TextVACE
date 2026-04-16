"""
FVD (Frechet Video Distance) computation utility.
Uses R3D-18 (Kinetics-400 pretrained) for video feature extraction.
"""

import numpy as np
import torch
from scipy.linalg import sqrtm


def get_r3d_model(device="cuda"):
    """Load pretrained R3D-18 model for feature extraction."""
    from torchvision.models.video import r3d_18, R3D_18_Weights
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1).to(device)
    model.eval()
    # Remove final classification layer → output 512-dim features
    model.fc = torch.nn.Identity()
    return model


def extract_video_features(model, video_frames, device="cuda", max_frames=16):
    """Extract R3D-18 features from a single video.

    Args:
        model: R3D-18 model (fc replaced with Identity)
        video_frames: list of RGB numpy arrays (H, W, 3)
        device: cuda or cpu
        max_frames: uniformly sample this many frames

    Returns:
        feature vector (512,)
    """
    import cv2

    n = len(video_frames)
    indices = np.linspace(0, n - 1, max_frames, dtype=int)
    frames = [video_frames[i] for i in indices]

    # Resize to 112x112 as expected by R3D-18
    resized = [cv2.resize(f, (112, 112)) for f in frames]
    tensor = np.stack(resized)  # (T, 112, 112, 3)
    tensor = torch.from_numpy(tensor).permute(3, 0, 1, 2).unsqueeze(0).float().to(device) / 255.0

    # Normalize with Kinetics-400 stats
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1).to(device)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1).to(device)
    tensor = (tensor - mean) / std

    with torch.no_grad():
        feat = model(tensor)  # (1, 512)

    return feat[0].cpu().numpy()


def compute_fvd_from_features(feats_real, feats_fake):
    """Compute FVD from two sets of feature vectors.

    Adds small regularization to covariance matrices for numerical
    stability when sample count < feature dimension.
    """
    mu_real = np.mean(feats_real, axis=0)
    mu_fake = np.mean(feats_fake, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    sigma_fake = np.cov(feats_fake, rowvar=False)

    # Regularize for numerical stability (important when N < dim)
    eps = 1e-6
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_fake += eps * np.eye(sigma_fake.shape[0])

    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fvd)


def compute_fvd_from_videos(edited_videos, gt_videos, device="cuda"):
    """Compute FVD between edited and GT video sets.

    Args:
        edited_videos: list of [list of RGB frames]
        gt_videos: list of [list of RGB frames]

    Returns:
        FVD score (float, lower is better)
    """
    model = get_r3d_model(device)

    print(f"  Extracting features from {len(edited_videos)} edited videos...")
    feats_edited = []
    for frames in edited_videos:
        feat = extract_video_features(model, frames, device)
        feats_edited.append(feat)

    print(f"  Extracting features from {len(gt_videos)} GT videos...")
    feats_gt = []
    for frames in gt_videos:
        feat = extract_video_features(model, frames, device)
        feats_gt.append(feat)

    feats_edited = np.stack(feats_edited)
    feats_gt = np.stack(feats_gt)

    fvd = compute_fvd_from_features(feats_gt, feats_edited)
    return fvd
