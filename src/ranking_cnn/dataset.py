"""
Improved Ranking Dataset – Balanced + Hard Negatives
"""

from __future__ import annotations

import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_generator import SyntheticWaferGenerator, GeneratorConfig
from src.classical import ClassicalPeakDetector, DetectorConfig


class RankingDataset(Dataset):
    def __init__(
        self,
        n_scenes: int = 400,
        patch_size: int = 64,
        top_k: int = 12,
        seed: int = 42,
    ):
        self.patch_size = patch_size
        self.samples: List[Tuple[np.ndarray, np.ndarray, float]] = []

        gen = SyntheticWaferGenerator(GeneratorConfig(seed=seed))
        detector = ClassicalPeakDetector(DetectorConfig(top_k=top_k))

        difficulties = ["easy", "medium", "hard"]
        styles = ["dram", "finfet"]

        print(f"Building balanced ranking dataset ({n_scenes} scenes) ...")

        for i in range(n_scenes):
            style = styles[i % 2]                    # strict 50/50
            difficulty = difficulties[i % 3]         # cycle easy/medium/hard

            pair = gen.generate_pair(style=style, difficulty=difficulty)
            ref = pair["reference"]
            search = pair["search"]
            gt = pair["gt_center"]

            # Scaled reference template
            th = max(8, ref.shape[0] // 10)
            tw = max(8, ref.shape[1] // 10)
            template = cv2.resize(ref, (tw, th), interpolation=cv2.INTER_AREA)

            peaks = detector.detect(ref, search)

            # ----- Positive -----
            pos_patch = self._crop(search, gt[0], gt[1])
            self.samples.append((pos_patch, template, 1.0))

            # ----- Hard Negatives -----
            # 1. Classical peaks that are NOT the GT
            for p in peaks:
                dist = np.hypot(p.x - gt[0], p.y - gt[1])
                if dist > 8.0:
                    neg_patch = self._crop(search, p.x, p.y)
                    self.samples.append((neg_patch, template, 0.0))

            # 2. Nearby offsets (very hard)
            for _ in range(2):
                ox = random.randint(4, 14) * random.choice([-1, 1])
                oy = random.randint(4, 14) * random.choice([-1, 1])
                neg_patch = self._crop(search, gt[0] + ox, gt[1] + oy)
                self.samples.append((neg_patch, template, 0.0))

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n_scenes}  (style={style}, diff={difficulty})")

        random.shuffle(self.samples)
        print(f"Total samples: {len(self.samples)}")

    def _crop(self, img: np.ndarray, cx: float, cy: float) -> np.ndarray:
        h, w = img.shape
        half = self.patch_size // 2
        x1 = int(round(cx)) - half
        y1 = int(round(cy)) - half

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x1 + self.patch_size - w)
        pad_bottom = max(0, y1 + self.patch_size - h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x1 + self.patch_size - pad_left)
        y2 = min(h, y1 + self.patch_size - pad_top)

        patch = img[y1:y2, x1:x2]
        if pad_left or pad_top or pad_right or pad_bottom:
            patch = cv2.copyMakeBorder(
                patch, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT_101
            )
        if patch.shape != (self.patch_size, self.patch_size):
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        return patch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cand, ref_tmpl, label = self.samples[idx]
        ref_resized = cv2.resize(ref_tmpl, (self.patch_size, self.patch_size),
                                 interpolation=cv2.INTER_AREA)

        x = np.stack([cand, ref_resized], axis=0).astype(np.float32) / 255.0
        y = np.array([label], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)