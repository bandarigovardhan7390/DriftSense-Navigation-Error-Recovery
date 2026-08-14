import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
    def forward(self, x):
        return self.net(x)

class SiameseRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Sequential(
            nn.Linear(128*3, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, ref, cand):
        f1 = self.encoder(ref)
        f2 = self.encoder(cand)
        x = torch.cat([f1, f2, torch.abs(f1-f2)], dim=1)
        return self.head(x)
    def score(self, ref, cand):
        return torch.sigmoid(self.forward(ref, cand))