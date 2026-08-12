#!/usr/bin/env python3
"""
Two-phase multi-task training for LMADGR (paper Sec. III).

Phase 1: λ_move = λ_sc = 1.0
Phase 2: λ_move = λ_sc = 0.1  (gesture-focused fine-tune)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from lmadgr.data import create_dataloaders
from lmadgr.losses import supervised_contrastive_loss
from lmadgr.models import ModifiedMambaVisionTwoClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_loaders(cfg: dict):
    data = cfg["data"]
    return create_dataloaders(
        root_dirs=[data["walking_dir"], data["still_dir"]],
        negative_root=data.get("motion_dir"),
        batch_size=data["batch_size"],
        val_split=data["val_split"],
        seed=cfg["train"]["seed"],
        num_workers=data["num_workers"],
        image_size=tuple(data["image_size"]),
        zero_out_percentage=data.get("zero_out_percentage", 0.0),
    )


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    gesture_loss_sum, correct, n = 0.0, 0, 0
    for inputs, labels, loco in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        g = model(inputs, gesture_only=True)
        gesture_loss_sum += criterion(g, labels).item() * labels.size(0)
        correct += (g.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return gesture_loss_sum / max(n, 1), 100.0 * correct / max(n, 1)


def run_phase(
    model,
    train_loader,
    val_loader,
    *,
    epochs: int,
    lr: float,
    lambda_move: float,
    lambda_sc: float,
    temperature: float,
    device,
    save_path: Path,
    phase_name: str,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=5)
    best_gesture_loss = float("inf")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"{phase_name} {epoch+1}/{epochs}")
        for inputs, labels, loco in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            loco = loco.to(device)
            optimizer.zero_grad()
            g, m, z = model(inputs)
            loss = (
                criterion(g, labels)
                + lambda_move * criterion(m, loco)
                + lambda_sc * supervised_contrastive_loss(z, labels, temperature)
            )
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (g.argmax(1) == labels).sum().item()
            n += labels.size(0)
            pbar.set_postfix(loss=f"{loss_sum/n:.4f}", acc=f"{100*correct/n:.2f}%")

        val_gesture_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_gesture_loss)
        print(
            f"{phase_name} epoch {epoch+1}: "
            f"train_acc={100*correct/n:.2f}% val_acc={val_acc:.2f}% val_gesture_loss={val_gesture_loss:.6f}"
        )
        if val_gesture_loss < best_gesture_loss:
            best_gesture_loss = val_gesture_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_gesture_loss": best_gesture_loss,
                    "lambda_move": lambda_move,
                    "lambda_sc": lambda_sc,
                },
                save_path,
            )
            print(f"  saved {save_path} (best val gesture loss {best_gesture_loss:.6f})")
    return best_gesture_loss


def main():
    parser = argparse.ArgumentParser(description="Train LMADGR (two-phase)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint for phase 2")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_loaders(cfg)
    model = ModifiedMambaVisionTwoClassifier(
        num_classes=cfg["data"]["num_gesture_classes"],
        num_classes_move=cfg["data"]["num_locomotion_classes"],
        model_name=cfg["model"]["model_name"],
        model_revision=cfg["model"].get("model_revision"),
        classifier_hidden_size=cfg["model"]["classifier_hidden_size"],
        locomotion_in_dim=cfg["model"]["locomotion_in_dim"],
        contrastive_dim=cfg["model"]["contrastive_dim"],
    ).to(device)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    best = Path(cfg["paths"]["best_ckpt"])
    tcfg = cfg["train"]

    if args.phase in ("1", "all"):
        phase1_path = ckpt_dir / "phase1_best.pth"
        run_phase(
            model,
            train_loader,
            val_loader,
            epochs=tcfg["phase1_epochs"],
            lr=tcfg["learning_rate"],
            lambda_move=tcfg["phase1_lambda_move"],
            lambda_sc=tcfg["phase1_lambda_sc"],
            temperature=tcfg["temperature"],
            device=device,
            save_path=phase1_path,
            phase_name="Phase1",
        )
        # Continue into phase 2 from best phase-1 weights
        state = torch.load(phase1_path, map_location=device)
        model.load_state_dict(state["model"])

    if args.phase in ("2", "all"):
        if args.phase == "2":
            resume = args.resume or str(ckpt_dir / "phase1_best.pth")
            state = torch.load(resume, map_location=device)
            model.load_state_dict(state["model"] if "model" in state else state)
            print(f"Loaded {resume}")
        run_phase(
            model,
            train_loader,
            val_loader,
            epochs=tcfg["phase2_epochs"],
            lr=tcfg["learning_rate"],
            lambda_move=tcfg["phase2_lambda_move"],
            lambda_sc=tcfg["phase2_lambda_sc"],
            temperature=tcfg["temperature"],
            device=device,
            save_path=best,
            phase_name="Phase2",
        )


if __name__ == "__main__":
    main()
