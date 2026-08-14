"""
Multi-Cue Classical Ranker
Combines intensity NCC, gradient similarity, SSIM, and larger-context score.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple
from .peak_detector import Peak


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a*a).sum() * (b*b).sum()) + 1e-8
    return float((a * b).sum() / denom)


def _gradient_mag(img: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight SSIM (single channel)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2*mu_a*mu_b + C1) * (2*sigma_ab + C2)) /
                 ((mu_a**2 + mu_b**2 + C1) * (sigma_a + sigma_b + C2) + 1e-8))


def _crop(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = img.shape
    half = size // 2
    x1 = max(0, min(int(round(cx)) - half, w - size))
    y1 = max(0, min(int(round(cy)) - half, h - size))
    patch = img[y1:y1+size, x1:x1+size]
    if patch.shape != (size, size):
        patch = cv2.resize(patch, (size, size))
    return patch


def rank_candidates(
    reference: np.ndarray,
    search: np.ndarray,
    peaks: List[Peak],
    small: int = 64,
    large: int = 128,
    weights: Tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20),
) -> List[Tuple[Peak, float]]:
    """
    Returns list of (peak, combined_score) sorted descending.
    weights = (ncc, gradient, ssim, context)
    """
    # Prepare reference templates
    ref_small = cv2.resize(reference, (small, small), interpolation=cv2.INTER_AREA)
    ref_large = cv2.resize(reference, (large, large), interpolation=cv2.INTER_AREA)
    ref_grad = _gradient_mag(ref_small)

    scored = []
    for p in peaks:
        # Intensity NCC (small)
        cand_s = _crop(search, p.x, p.y, small)
        ncc = _ncc(ref_small, cand_s)

        # Gradient similarity
        cand_g = _gradient_mag(cand_s)
        grad = _ncc(ref_grad, cand_g)

        # SSIM
        ssim = _ssim_simple(ref_small, cand_s)

        # Larger context
        cand_l = _crop(search, p.x, p.y, large)
        ctx = _ncc(ref_large, cand_l)

        # Combined
        w1, w2, w3, w4 = weights
        score = w1*ncc + w2*grad + w3*ssim + w4*ctx
        scored.append((p, float(score)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored