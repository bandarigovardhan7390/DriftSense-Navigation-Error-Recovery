#!/usr/bin/env python3
"""
Drift-Sense – Standalone Localization Inference Script
======================================================
CRITICAL FILE for Applied Materials evaluation.

Usage:
    python scripts/inference.py <reference_image> <search_image>

Example:
    python scripts/inference.py ref.png search.png

Output:
    Prints a single line:  x,y
    (the predicted centre of the reference pattern inside the search image)

This script must run without manual edits on a fresh machine.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Allow running from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classical import ClassicalPeakDetector, DetectorConfig
from src.ranking_cnn.siamese_model import SiameseRanker

# ---------------------------------------------------------------------------
# Configuration (locked)
# ---------------------------------------------------------------------------
PATCH_SIZE = 96
CLASSICAL_CONF_THRESHOLD = 0.85
MODEL_CANDIDATES = [
    ROOT / "models" / "siamese_final_96.pt",
    ROOT / "outputs" / "models" / "siamese_final_96.pt",
]


def load_image(path: str | Path) -> np.ndarray:
    """Load image as grayscale uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # fallback with Pillow
        from PIL import Image
        img = np.array(Image.open(path).convert("L"))
    if img is None or img.size == 0:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def crop(img: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = img.shape
    half = size // 2
    x1 = max(0, min(int(round(cx)) - half, w - size))
    y1 = max(0, min(int(round(cy)) - half, h - size))
    patch = img[y1:y1 + size, x1:x1 + size]
    if patch.shape != (size, size):
        patch = cv2.resize(patch, (size, size))
    return patch


def predict(reference: np.ndarray, search: np.ndarray, model, device) -> tuple[float, float]:
    """
    Core localization logic.
    Returns (x, y) centre coordinates.
    """
    detector = ClassicalPeakDetector(DetectorConfig(top_k=20))
    peaks = detector.detect(reference, search)

    if not peaks:
        # fallback to image centre
        h, w = search.shape
        return w / 2.0, h / 2.0

    # High classical confidence → trust classical
    if peaks[0].score > CLASSICAL_CONF_THRESHOLD:
        return float(peaks[0].x), float(peaks[0].y)

    # Otherwise rank with Siamese
    if model is None:
        return float(peaks[0].x), float(peaks[0].y)

    ps = PATCH_SIZE
    tmpl = cv2.resize(reference, (ps, ps)).astype(np.float32) / 255.0
    ref_t = torch.from_numpy(tmpl[None, None]).to(device)

    best_score = -1.0
    best_peak = peaks[0]

    model.eval()
    with torch.no_grad():
        for p in peaks:
            cand = crop(search, p.x, p.y, ps).astype(np.float32) / 255.0
            cand_t = torch.from_numpy(cand[None, None]).to(device)
            s = float(model.score(ref_t, cand_t).cpu())
            if s > best_score:
                best_score = s
                best_peak = p

    return float(best_peak.x), float(best_peak.y)


def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense localization inference. Outputs predicted centre x,y"
    )
    parser.add_argument("reference", type=str, help="Path to reference image")
    parser.add_argument("search", type=str, help="Path to search image")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if omitted)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = None
    model_path = None
    for cand in MODEL_CANDIDATES:
        if cand.exists():
            model_path = cand
            break

    if model_path is not None:
        model = SiameseRanker().to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()
    # else: classical-only fallback

    # Load images
    ref = load_image(args.reference)
    search = load_image(args.search)

    # Predict
    x, y = predict(ref, search, model, device)

    # REQUIRED OUTPUT FORMAT: single line "x,y"
    print(f"{x:.2f},{y:.2f}")


if __name__ == "__main__":
    main()
