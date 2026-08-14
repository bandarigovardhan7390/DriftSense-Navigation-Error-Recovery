"""
Lightweight Ranking CNN (64×64) – FPGA friendly
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RankingCNN(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 32×32

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),            # 1×1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = RankingCNN()
    print(model)
    print(f"Parameters: {count_parameters(model):,}")
    dummy = torch.randn(2, 2, 64, 64)
    print("Output shape:", model(dummy).shape)