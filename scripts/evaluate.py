#!/usr/bin/env python3
"""Evaluate a trained LMADGR checkpoint (gesture + locomotion metrics)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from lmadgr.data import GESTURE_CLASSES, LOCOMOTION_CLASSES, create_dataloaders
from lmadgr.models import ModifiedMambaVisionTwoClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = cfg["data"]
    _, val_loader = create_dataloaders(
        root_dirs=[data["walking_dir"], data["still_dir"]],
        negative_root=data.get("motion_dir"),
        batch_size=data["batch_size"],
        val_split=data["val_split"],
        seed=cfg["train"]["seed"],
        num_workers=data["num_workers"],
        image_size=tuple(data["image_size"]),
        zero_out_percentage=0.0,
    )

    model = ModifiedMambaVisionTwoClassifier(
        num_classes=data["num_gesture_classes"],
        num_classes_move=data["num_locomotion_classes"],
        model_name=cfg["model"]["model_name"],
        classifier_hidden_size=cfg["model"]["classifier_hidden_size"],
        locomotion_in_dim=cfg["model"]["locomotion_in_dim"],
        contrastive_dim=cfg["model"]["contrastive_dim"],
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    g_preds, g_labels, m_preds, m_labels = [], [], [], []
    with torch.no_grad():
        for inputs, labels, loco in tqdm(val_loader, desc="Eval"):
            g, m, _ = model(inputs.to(device))
            g_preds.extend(g.argmax(1).cpu().tolist())
            g_labels.extend(labels.tolist())
            m_preds.extend(m.argmax(1).cpu().tolist())
            m_labels.extend(loco.tolist())

    g_names = list(GESTURE_CLASSES.keys())
    m_names = [LOCOMOTION_CLASSES[i] for i in range(len(LOCOMOTION_CLASSES))]
    acc = 100.0 * np.mean(np.array(g_preds) == np.array(g_labels))
    print(f"Gesture accuracy: {acc:.2f}%")
    print(classification_report(g_labels, g_preds, target_names=g_names, digits=4))
    print("Locomotion:")
    print(classification_report(m_labels, m_preds, target_names=m_names, digits=4))
    print(f"Gesture macro-F1: {f1_score(g_labels, g_preds, average='macro'):.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "cm_gesture.npy", confusion_matrix(g_labels, g_preds))
    np.save(args.out_dir / "cm_locomotion.npy", confusion_matrix(m_labels, m_preds))
    print(f"Saved confusion matrices to {args.out_dir}")


if __name__ == "__main__":
    main()
