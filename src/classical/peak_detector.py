"""
Drift-Sense Classical Peak Detector
===================================

Stage-1 of the hybrid pipeline:
1. Down-scale the Reference by exactly 10× → ~100×100 template
2. Run normalized cross-correlation against the Search image
3. Extract the Top-K peaks (with non-maximum suppression)
4. Return candidate centres + scores

Author: Drift-Sense Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Peak:
    x: float
    y: float
    score: float
    rank: int = 0


@dataclass
class DetectorConfig:
    scale_factor: int = 10
    top_k: int = 20
    nms_radius: int = 12
    method: str = "ncc"
    min_score: float = 0.15


class ClassicalPeakDetector:
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()

    def detect(self, reference: np.ndarray, search: np.ndarray) -> List[Peak]:
        assert reference.ndim == 2 and search.ndim == 2

        th = max(8, reference.shape[0] // self.cfg.scale_factor)
        tw = max(8, reference.shape[1] // self.cfg.scale_factor)
        template = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)

        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        peaks = self._extract_peaks(result, template.shape)
        return peaks

    def _extract_peaks(self, corr_map: np.ndarray, template_shape: Tuple[int, int]) -> List[Peak]:
        th, tw = template_shape
        cmap = corr_map.copy()
        peaks: List[Peak] = []

        for rank in range(self.cfg.top_k):
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cmap)
            if max_val < self.cfg.min_score:
                break

            cx = max_loc[0] + tw / 2.0
            cy = max_loc[1] + th / 2.0
            peaks.append(Peak(x=cx, y=cy, score=float(max_val), rank=rank))

            x0 = max(0, max_loc[0] - self.cfg.nms_radius)
            y0 = max(0, max_loc[1] - self.cfg.nms_radius)
            x1 = min(cmap.shape[1], max_loc[0] + self.cfg.nms_radius + 1)
            y1 = min(cmap.shape[0], max_loc[1] + self.cfg.nms_radius + 1)
            cmap[y0:y1, x0:x1] = -1.0

        return peaks

    @staticmethod
    def is_hit(peaks: List[Peak], gt_center: Tuple[float, float],
               tolerance_px: float = 5.0, k: Optional[int] = None) -> bool:
        k = k or len(peaks)
        gx, gy = gt_center
        for p in peaks[:k]:
            if np.hypot(p.x - gx, p.y - gy) <= tolerance_px:
                return True
        return False

    @staticmethod
    def best_distance(peaks: List[Peak], gt_center: Tuple[float, float],
                      k: Optional[int] = None) -> float:
        k = k or len(peaks)
        if not peaks:
            return float("inf")
        gx, gy = gt_center
        return min(np.hypot(p.x - gx, p.y - gy) for p in peaks[:k])