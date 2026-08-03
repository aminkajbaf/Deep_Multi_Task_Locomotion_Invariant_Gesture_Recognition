"""Supervised contrastive (InfoNCE-style) loss with gesture-only positives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Args:
        features: (B, D) projection-head outputs
        labels:   (B,) gesture class ids
        temperature: softmax temperature τ
    """
    features = F.normalize(features, dim=1)
    sim = torch.matmul(features, features.T) / temperature

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    mask = mask - torch.eye(mask.size(0), device=mask.device)

    positives = mask.sum(dim=1)
    log_prob = F.log_softmax(sim, dim=1)
    loss = -(mask * log_prob).sum(dim=1) / (positives + 1e-8)
    return loss.mean()
