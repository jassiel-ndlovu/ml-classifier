"""The GlyphCNN architecture.

A compact 3-block convolutional network (Conv-BN-ReLU-Pool with dropout) followed
by a fully-connected head. The layer shapes are kept identical to the original
project so that the historical ``pretrained_model.pth`` checkpoint loads directly,
while the dropout rate is exposed through the constructor for experimentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IMAGE_SIZE


class GlyphCNN(nn.Module):
    """Input: (N, 1, 32, 32) float tensor. Output: (N, num_classes) logits."""

    def __init__(self, num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        feat = IMAGE_SIZE // 8  # three 2x2 pools: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(128 * feat * feat, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(min(0.5, dropout * 2))
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.pool(self.bn1(F.relu(self.conv1(x)))))
        x = self.dropout2(self.pool(self.bn2(F.relu(self.conv2(x)))))
        x = self.dropout3(self.pool(self.bn3(F.relu(self.conv3(x)))))
        x = torch.flatten(x, 1)
        x = self.dropout4(F.relu(self.bn4(self.fc1(x))))
        return self.fc2(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities in eval mode."""
        was_training = self.training
        self.eval()
        out = F.softmax(self.forward(x), dim=1)
        if was_training:
            self.train()
        return out
