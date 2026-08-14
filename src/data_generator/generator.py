"""
Drift-Sense Synthetic Data Generator  (v5 – Classical Trap support)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class GeneratorConfig:
    image_size: int = 1000
    scale_factor: int = 10
    dram_period_x: int = 28
    dram_period_y: int = 28
    dram_contact_radius: int = 6
    dram_line_width: int = 5
    fin_period: int = 18
    fin_width: int = 7
    gate_period: int = 42
    gate_width: int = 11
    seed: Optional[int] = None


class SyntheticWaferGenerator:
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.cfg = config or GeneratorConfig()
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)
            random.seed(self.cfg.seed)

    def generate_pair(self, style: str = "dram", difficulty: str = "medium") -> Dict:
        style = style.lower()
        difficulty = difficulty.lower()
        assert style in ("dram", "finfet")
        assert difficulty in ("easy", "medium", "hard", "adversarial", "trap")

        # Difficulty parameters
        params = {
            "easy":        dict(rot=0.4, scale=0.01, noise=0.5, uniq=0.25, blur=0.4, contrast=0.07, alpha=0.95, trap=False),
            "medium":      dict(rot=1.2, scale=0.02, noise=0.9, uniq=0.38, blur=0.8, contrast=0.14, alpha=0.92, trap=False),
            "hard":        dict(rot=2.5, scale=0.04, noise=1.4, uniq=0.50, blur=1.2, contrast=0.22, alpha=0.85, trap=False),
            "adversarial": dict(rot=3.5, scale=0.06, noise=1.8, uniq=0.60, blur=1.5, contrast=0.28, alpha=0.78, trap=False),
            "trap":        dict(rot=1.8, scale=0.03, noise=1.1, uniq=0.45, blur=1.0, contrast=0.18, alpha=0.72, trap=True),
        }[difficulty]

        S = self.cfg.image_size
        F = self.cfg.scale_factor

        tile_size = 2200
        tile = self._create_layout(tile_size, style, uniqueness=params["uniq"])

        margin = 550
        cx = random.randint(margin, tile_size - margin)
        cy = random.randint(margin, tile_size - margin)

        angle = random.uniform(-params["rot"], params["rot"])
        scale = random.uniform(1.0 - params["scale"], 1.0 + params["scale"])

        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        tile = cv2.warpAffine(tile, M, (tile_size, tile_size),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        pts = np.array([[[cx, cy]]], dtype=np.float32)
        new_pt = cv2.transform(pts, M)[0, 0]
        cx, cy = float(new_pt[0]), float(new_pt[1])

        # Reference
        half = S // 2
        y0 = int(cy) - half
        x0 = int(cx) - half
        ref_hr = tile[y0:y0+S, x0:x0+S].copy()
        if ref_hr.shape != (S, S):
            ref_hr = cv2.resize(ref_hr, (S, S), interpolation=cv2.INTER_AREA)

        # Search background
        bg = cv2.resize(tile, (S, S), interpolation=cv2.INTER_AREA)
        template = cv2.resize(ref_hr, (S // F, S // F), interpolation=cv2.INTER_AREA)
        th, tw = template.shape

        # True location
        max_x = S - tw - 40
        max_y = S - th - 40
        gt_x = random.randint(50, max_x) + tw / 2.0
        gt_y = random.randint(50, max_y) + th / 2.0

        search = bg.copy()

        # Insert true pattern (slightly degraded in trap mode)
        x1 = int(gt_x - tw / 2)
        y1 = int(gt_y - th / 2)
        alpha = params["alpha"]
        search[y1:y1+th, x1:x1+tw] = (
            alpha * template + (1 - alpha) * search[y1:y1+th, x1:x1+tw]
        )

        # ===== CLASSICAL TRAP: insert a stronger competing distractor =====
        if params["trap"]:
            # Create a cleaner version of the template as distractor
            clean_template = cv2.GaussianBlur(template, (0, 0), 0.3)
            clean_template = np.clip(clean_template * 1.08, 0, 255)

            # Place distractor at a nearby periodic offset or random location
            period = 28 if style == "dram" else 18
            offset = period * random.choice([2, 3, 4, -2, -3])
            if random.random() > 0.5:
                dx, dy = offset, random.randint(-8, 8)
            else:
                dx, dy = random.randint(-8, 8), offset

            dist_x = int(np.clip(gt_x + dx, tw//2 + 20, S - tw//2 - 20))
            dist_y = int(np.clip(gt_y + dy, th//2 + 20, S - th//2 - 20))

            dx1 = int(dist_x - tw / 2)
            dy1 = int(dist_y - th / 2)

            # Higher alpha → stronger correlation peak (classical prefers this)
            search[dy1:dy1+th, dx1:dx1+tw] = (
                0.93 * clean_template + 0.07 * search[dy1:dy1+th, dx1:dx1+tw]
            )

        del tile

        # Noise
        reference = self._add_sem_noise(ref_hr, is_reference=True, mult=params)
        search = self._add_sem_noise(search, is_reference=False, mult=params)

        return {
            "reference": reference,
            "search": search,
            "gt_center": (float(gt_x), float(gt_y)),
            "style": style,
            "difficulty": difficulty,
            "meta": {"rotation_deg": angle, "scale": scale},
        }

    # ------------------------------------------------------------------
    def _create_layout(self, size: int, style: str, uniqueness: float = 0.4) -> np.ndarray:
        canvas = np.zeros((size, size), dtype=np.float32)
        if style == "dram":
            self._draw_dram(canvas)
        else:
            self._draw_finfet(canvas)
        self._add_uniqueness(canvas, strength=uniqueness)
        return self._normalize(canvas)

    def _draw_dram(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape
        px, py = self.cfg.dram_period_x, self.cfg.dram_period_y
        lw, r = self.cfg.dram_line_width, self.cfg.dram_contact_radius
        for y in range(0, h, py):
            canvas[y:y+lw, :] = 175
        for x in range(0, w, px):
            canvas[:, x:x+lw] = 155
        for y in range(py//2, h, py):
            for x in range(px//2, w, px):
                cv2.circle(canvas, (x, y), r, 225, -1)

    def _draw_finfet(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape
        fp, fw = self.cfg.fin_period, self.cfg.fin_width
        gp, gw = self.cfg.gate_period, self.cfg.gate_width
        for x in range(0, w, fp):
            canvas[:, x:x+fw] = 170
        for y in range(0, h, gp):
            canvas[y:y+gw, :] = 205

    def _add_uniqueness(self, canvas: np.ndarray, strength: float = 0.4) -> None:
        if strength <= 0:
            return
        h, w = canvas.shape
        for _ in range(int(15 + 35 * strength)):
            x = random.randint(30, w-30)
            y = random.randint(30, h-30)
            rad = random.randint(3, 11)
            intens = random.uniform(10, 40) * strength
            sign = 1 if random.random() > 0.35 else -1
            cv2.circle(canvas, (x, y), rad, float(sign * intens), -1)

    def _add_sem_noise(self, img: np.ndarray, is_reference: bool, mult: dict) -> np.ndarray:
        img = img.astype(np.float32)
        h, w = img.shape
        yy, xx = np.mgrid[0:h, 0:w]
        amp = 0.12 * mult["noise"]
        illum = 1.0 + amp * (
            0.55 * np.sin(2 * math.pi * xx / w * random.uniform(0.3, 1.7))
            + 0.45 * np.sin(2 * math.pi * yy / h * random.uniform(0.3, 1.7))
        )
        img *= illum
        sigma = 0.6 * mult["blur"] * (0.7 if is_reference else 1.2)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=max(0.3, sigma))
        noise_std = 5.0 * mult["noise"] * (0.6 if is_reference else 1.1)
        img += np.random.normal(0.0, noise_std, img.shape)
        contrast = 1.0 + random.uniform(-mult["contrast"], mult["contrast"])
        mean = float(img.mean())
        img = mean + (img - mean) * contrast
        return np.clip(img, 0, 255).astype(np.uint8)

    def _normalize(self, img: np.ndarray, low=30, high=220) -> np.ndarray:
        img = img - img.min()
        if img.max() > 1e-5:
            img = img / img.max()
        return low + img * (high - low)