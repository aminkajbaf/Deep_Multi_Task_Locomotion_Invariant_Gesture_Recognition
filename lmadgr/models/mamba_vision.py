"""
Modified MambaVision encoder for 150-channel radar RAP tensors (paper Sec. III).

Heads:
  - Gesture classifier (main task)
  - Locomotion / movement classifier (auxiliary)
  - Supervised-contrastive projection head
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel


def _adapt_patch_embed(model: nn.Module, in_channels: int = 150) -> None:
    original = model.model.patch_embed.conv_down[0]
    model.model.patch_embed.conv_down[0] = nn.Conv2d(
        in_channels,
        original.out_channels,
        kernel_size=original.kernel_size,
        stride=original.stride,
        padding=original.padding,
        bias=original.bias is not None,
    )


class ModifiedMambaVision(nn.Module):
    """Gesture-only baseline (ablation)."""

    def __init__(
        self,
        num_classes: int = 13,
        model_name: str = "nvidia/MambaVision-L-21K",
        classifier_hidden_size: int = 1568,
        in_channels: int = 150,
    ):
        super().__init__()
        self.mamba_vision = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _adapt_patch_embed(self.mamba_vision, in_channels)
        self.classifier = nn.Linear(classifier_hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.mamba_vision(x)
        return self.classifier(h)


class ModifiedMambaVisionMovementOnly(nn.Module):
    """Gesture + locomotion heads (no contrastive)."""

    def __init__(
        self,
        num_classes: int = 13,
        num_classes_move: int = 3,
        model_name: str = "nvidia/MambaVision-L-21K",
        classifier_hidden_size: int = 1568,
        locomotion_in_dim: int = 47040,
        in_channels: int = 150,
    ):
        super().__init__()
        self.mamba_vision = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _adapt_patch_embed(self.mamba_vision, in_channels)
        self.classifier = nn.Linear(classifier_hidden_size, num_classes)
        self.classifier_move = nn.Sequential(
            nn.Linear(locomotion_in_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes_move),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, features = self.mamba_vision(x)
        gesture = self.classifier(h)
        flat = torch.cat([f.flatten(1) for f in features], dim=1)
        move = self.classifier_move(flat)
        return gesture, move


class ModifiedMambaVisionContrastiveOnly(nn.Module):
    """Gesture + supervised-contrastive projection (no locomotion)."""

    def __init__(
        self,
        num_classes: int = 13,
        model_name: str = "nvidia/MambaVision-L-21K",
        classifier_hidden_size: int = 1568,
        contrastive_dim: int = 1024,
        in_channels: int = 150,
    ):
        super().__init__()
        self.mamba_vision = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _adapt_patch_embed(self.mamba_vision, in_channels)
        self.classifier = nn.Linear(classifier_hidden_size, num_classes)
        self.contrastive_projection = nn.Sequential(
            nn.Linear(classifier_hidden_size, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, contrastive_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.mamba_vision(x)
        return self.classifier(h), self.contrastive_projection(h)


class ModifiedMambaVisionTwoClassifier(nn.Module):
    """Full LMADGR model: gesture + locomotion + supervised contrastive."""

    def __init__(
        self,
        num_classes: int = 13,
        num_classes_move: int = 3,
        model_name: str = "nvidia/MambaVision-L-21K",
        classifier_hidden_size: int = 1568,
        locomotion_in_dim: int = 47040,
        contrastive_dim: int = 1024,
        in_channels: int = 150,
    ):
        super().__init__()
        self.mamba_vision = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _adapt_patch_embed(self.mamba_vision, in_channels)
        self.classifier = nn.Linear(classifier_hidden_size, num_classes)
        self.classifier_move = nn.Sequential(
            nn.Linear(locomotion_in_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes_move),
        )
        self.contrastive_projection = nn.Sequential(
            nn.Linear(classifier_hidden_size, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, contrastive_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, features = self.mamba_vision(x)
        gesture_logits = self.classifier(h)
        flat = torch.cat([f.flatten(1) for f in features], dim=1)
        move_logits = self.classifier_move(flat)
        contrastive = self.contrastive_projection(h)
        return gesture_logits, move_logits, contrastive
