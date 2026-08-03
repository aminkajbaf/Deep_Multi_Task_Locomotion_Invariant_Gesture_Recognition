"""Baseline classifiers for LAGID comparisons (150-channel RAP input)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DIGestureBaseline(nn.Module):
    """CNN + LSTM over spatial tokens (DI-Gesture family, adapted to RAP stacks)."""

    def __init__(self, num_classes: int = 13, in_channels: int = 150):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 128, 4, 4) -> (B, 16, 128) spatial sequence
        f = self.conv(x).flatten(2).transpose(1, 2)
        out, _ = self.lstm(f)
        return self.fc(out[:, -1])


class MLFFBaseline(nn.Module):
    """Compact multi-scale CNN classifier for 150-channel RAP."""

    def __init__(self, num_classes: int = 13, in_channels: int = 150):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x).flatten(1))


__all__ = ["DIGestureBaseline", "MLFFBaseline"]
