#!/usr/bin/env python3
"""Ablation trainers: baseline / +movement / +contrastive / full."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from lmadgr.data import create_dataloaders
from lmadgr.losses import supervised_contrastive_loss
from lmadgr.models import (
    ModifiedMambaVision,
    ModifiedMambaVisionContrastiveOnly,
    ModifiedMambaVisionMovementOnly,
    ModifiedMambaVisionTwoClassifier,
)


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(variant: str, cfg: dict) -> nn.Module:
    common = dict(
        num_classes=cfg["data"]["num_gesture_classes"],
        model_name=cfg["model"]["model_name"],
        model_revision=cfg["model"].get("model_revision"),
        classifier_hidden_size=cfg["model"]["classifier_hidden_size"],
    )
    if variant == "baseline":
        return ModifiedMambaVision(**common)
    if variant == "movement":
        return ModifiedMambaVisionMovementOnly(
            **common,
            num_classes_move=cfg["data"]["num_locomotion_classes"],
            locomotion_in_dim=cfg["model"]["locomotion_in_dim"],
        )
    if variant == "contrastive":
        return ModifiedMambaVisionContrastiveOnly(
            **common, contrastive_dim=cfg["model"]["contrastive_dim"]
        )
    if variant == "full":
        return ModifiedMambaVisionTwoClassifier(
            **common,
            num_classes_move=cfg["data"]["num_locomotion_classes"],
            locomotion_in_dim=cfg["model"]["locomotion_in_dim"],
            contrastive_dim=cfg["model"]["contrastive_dim"],
        )
    raise ValueError(variant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["baseline", "movement", "contrastive", "full"],
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lambda-aux", type=float, default=0.1)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = cfg["data"]
    train_loader, val_loader = create_dataloaders(
        root_dirs=[data["walking_dir"], data["still_dir"]],
        negative_root=data.get("motion_dir"),
        batch_size=data["batch_size"],
        val_split=data["val_split"],
        seed=cfg["train"]["seed"],
        num_workers=data["num_workers"],
        image_size=tuple(data["image_size"]),
        zero_out_percentage=data.get("zero_out_percentage", 0.0),
    )

    model = build_model(args.variant, cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)
    save_path = Path(cfg["paths"]["checkpoint_dir"]) / f"best_{args.variant}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    temp = cfg["train"]["temperature"]
    lam = args.lambda_aux

    for epoch in range(args.epochs):
        model.train()
        correct, n = 0, 0
        for inputs, labels, loco in tqdm(train_loader, desc=f"{args.variant} {epoch+1}"):
            inputs, labels, loco = inputs.to(device), labels.to(device), loco.to(device)
            optimizer.zero_grad()
            if args.variant == "baseline":
                g = model(inputs)
                loss = criterion(g, labels)
            elif args.variant == "movement":
                g, m = model(inputs)
                loss = criterion(g, labels) + lam * criterion(m, loco)
            elif args.variant == "contrastive":
                g, z = model(inputs)
                loss = criterion(g, labels) + lam * supervised_contrastive_loss(z, labels, temp)
            else:
                g, m, z = model(inputs)
                loss = (
                    criterion(g, labels)
                    + lam * criterion(m, loco)
                    + lam * supervised_contrastive_loss(z, labels, temp)
                )
            loss.backward()
            optimizer.step()
            correct += (g.argmax(1) == labels).sum().item()
            n += labels.size(0)

        model.eval()
        v_correct, v_n = 0, 0
        v_loss_sum = 0.0
        with torch.no_grad():
            for inputs, labels, loco in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if args.variant == "baseline":
                    g = model(inputs)
                elif args.variant == "movement":
                    g, _ = model(inputs)
                elif args.variant == "contrastive":
                    g, _ = model(inputs)
                else:
                    g = model(inputs, gesture_only=True)
                v_correct += (g.argmax(1) == labels).sum().item()
                v_n += labels.size(0)
                v_loss_sum += criterion(g, labels).item() * labels.size(0)

        val_acc = 100.0 * v_correct / max(v_n, 1)
        val_gesture_loss = v_loss_sum / max(v_n, 1)
        scheduler.step(val_gesture_loss)
        print(
            f"epoch {epoch+1}: train={100*correct/n:.2f}% val={val_acc:.2f}% "
            f"val_gesture_loss={val_gesture_loss:.6f}"
        )
        if val_gesture_loss < best:
            best = val_gesture_loss
            torch.save(
                {"model": model.state_dict(), "val_gesture_loss": best},
                save_path,
            )
            print(f"  saved {save_path}")


if __name__ == "__main__":
    main()
