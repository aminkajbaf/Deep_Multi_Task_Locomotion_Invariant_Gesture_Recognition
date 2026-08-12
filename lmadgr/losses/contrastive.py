"""Supervised contrastive loss for LMADGR."""

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
    logits = torch.matmul(features, features.T) / temperature

    bsz = logits.size(0)
    device = logits.device
    diag = torch.eye(bsz, dtype=torch.bool, device=device)

    # Remove i from the denominator (A(i) = {1..B} \ {i})
    logits = logits.masked_fill(diag, float("-inf"))
    log_prob = F.log_softmax(logits, dim=1)

    labels = labels.view(-1)
    pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~diag
    num_pos = pos_mask.sum(dim=1)

    loss_i = -(pos_mask * log_prob).sum(dim=1)
    valid = num_pos > 0
    loss_i = torch.where(valid, loss_i / num_pos.clamp_min(1), torch.zeros_like(loss_i))

    if valid.any():
        return loss_i[valid].mean()
    return torch.zeros((), device=device, dtype=logits.dtype)
