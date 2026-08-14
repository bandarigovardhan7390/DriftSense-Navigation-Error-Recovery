"""
Drift-Sense Final Hybrid Pipeline
Classical NCC Top-20 → Siamese 96×96 ranking → safety fallback
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import cv2
import numpy as np
import torch

from src.classical import ClassicalPeakDetector, DetectorConfig
from src.ranking_cnn.siamese_model import SiameseRanker


class HybridPipeline:
    def __init__(
        self,
        model_path: str | Path | None = None,
        top_k: int = 20,
        patch_size: int = 96,
        device: str = "cpu",
        use_cnn: bool = True,
        cnn_confidence_threshold: float = 0.25,
    ):
        self.device = device
        self.patch_size = patch_size
        self.use_cnn = use_cnn
        self.cnn_confidence_threshold = cnn_confidence_threshold

        self.detector = ClassicalPeakDetector(DetectorConfig(top_k=top_k))

        self.model = None
        if use_cnn and model_path is not None:
            self.model = SiameseRanker().to(device)
            state = torch.load(model_path, map_location=device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
            self.model.eval()

    def _crop(self, img: np.ndarray, cx: float, cy: float) -> np.ndarray:
        h, w = img.shape
        half = self.patch_size // 2
        x1 = max(0, min(int(round(cx)) - half, w - self.patch_size))
        y1 = max(0, min(int(round(cy)) - half, h - self.patch_size))
        patch = img[y1:y1+self.patch_size, x1:x1+self.patch_size]
        if patch.shape != (self.patch_size, self.patch_size):
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        return patch

    @torch.no_grad()
    def predict(self, reference: np.ndarray, search: np.ndarray) -> Dict:
        peaks = self.detector.detect(reference, search)
        if not peaks:
            h, w = search.shape
            return {"x": w/2.0, "y": h/2.0, "score": 0.0, "method": "fallback"}

        # Default classical
        best = peaks[0]
        method = "classical"
        cnn_scores = []

        if self.use_cnn and self.model is not None:
            tmpl = cv2.resize(reference, (self.patch_size, self.patch_size)).astype(np.float32) / 255
            ref_t = torch.from_numpy(tmpl[None, None]).to(self.device)

            for p in peaks:
                cand = self._crop(search, p.x, p.y).astype(np.float32) / 255
                cand_t = torch.from_numpy(cand[None, None]).to(self.device)
                s = float(self.model.score(ref_t, cand_t).cpu())
                cnn_scores.append(s)

            max_s = max(cnn_scores) if cnn_scores else 0.0
            if max_s >= self.cnn_confidence_threshold:
                best_idx = int(np.argmax(cnn_scores))
                best = peaks[best_idx]
                method = "siamese"

        return {
            "x": float(best.x),
            "y": float(best.y),
            "score": float(best.score) if method == "classical" else (max(cnn_scores) if cnn_scores else 0.0),
            "method": method,
            "peaks": peaks,
            "cnn_scores": cnn_scores,
        }