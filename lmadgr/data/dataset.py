"""
LAGID-style dual-label dataset (gesture + locomotion).

Locomotion labels:
  0 = walking, 1 = still, 2 = non-gesture body motion
"""

from __future__ import annotations

import os
import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

GESTURE_CLASSES = {
    "Swipe to Right": 0,
    "Swipe to Left": 1,
    "Pull": 2,
    "Chek": 3,
    "Double Push": 4,
    "Rotate CCW": 5,
    "Rotate CW": 6,
    "Moving Finger": 7,
    "Double Hand Push": 8,
    "Cross": 9,
    "Swipe to Forward and Backward": 10,
    "Push": 11,
    "Motion": 12,
}

LOCOMOTION_CLASSES = {0: "Walking", 1: "Still", 2: "Motion"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _parse_gesture(path: str) -> int:
    basename = os.path.basename(path).replace(".npy", "")
    if "Motion" in basename:
        return GESTURE_CLASSES["Motion"]
    gesture = basename.split("_", 2)[-1] if basename.count("_") >= 2 else basename
    # Try split('_')[2] convention used in original code
    parts = basename.split("_")
    if len(parts) >= 3:
        candidate = parts[2]
        if candidate in GESTURE_CLASSES:
            return GESTURE_CLASSES[candidate]
        # Full remainder may contain spaces: prefix_id_Swipe to Right
        remainder = basename.split("_", 2)[2]
        if remainder in GESTURE_CLASSES:
            return GESTURE_CLASSES[remainder]
    if gesture in GESTURE_CLASSES:
        return GESTURE_CLASSES[gesture]
    raise KeyError(f"Cannot parse gesture from: {path}")


class GestureDatasetTwoClass(Dataset):
    def __init__(
        self,
        root_dirs: Sequence[str],
        negative_root: Optional[str] = None,
        transform: Optional[Callable] = None,
        zero_out_percentage: float = 0.0,
    ):
        """
        Args:
            root_dirs: [walking_dir, still_dir] — order defines locomotion ids 0, 1
            negative_root: non-gesture motion folder → locomotion id 2
        """
        self.transform = transform
        self.zero_out_percentage = zero_out_percentage
        self.classes = GESTURE_CLASSES

        pairs: List[Tuple[str, int]] = []
        for loco_id, root in enumerate(root_dirs):
            for name in os.listdir(root):
                if name.endswith(".npy"):
                    pairs.append((os.path.join(root, name), loco_id))
        if negative_root:
            loco_id = len(root_dirs)
            for name in os.listdir(negative_root):
                if name.endswith(".npy"):
                    pairs.append((os.path.join(negative_root, name), loco_id))

        # Shuffle paths and locomotion labels together
        random.shuffle(pairs)
        self.file_paths = [p for p, _ in pairs]
        self.locomotion = [l for _, l in pairs]
        self.labels = [_parse_gesture(p) for p in self.file_paths]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        data = torch.from_numpy(np.load(self.file_paths[idx]).astype(np.float32))
        dmin, dmax = data.min(), data.max()
        data = (data - dmin) / (dmax - dmin + 1e-8)
        if self.zero_out_percentage > 0:
            data = data * (torch.rand_like(data) > self.zero_out_percentage).float()
        if self.transform is not None:
            data = self.transform(data)
        return data, self.labels[idx], self.locomotion[idx]


def create_dataloaders(
    root_dirs: Sequence[str],
    negative_root: Optional[str] = None,
    batch_size: int = 8,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (32, 64),
    zero_out_percentage: float = 0.0,
) -> Tuple[DataLoader, DataLoader]:
    set_seed(seed)
    transform = transforms.Resize(list(image_size))
    full = GestureDatasetTwoClass(
        root_dirs=root_dirs,
        negative_root=negative_root,
        transform=transform,
        zero_out_percentage=zero_out_percentage,
    )
    val_size = int(len(full) * val_split)
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(
        full, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
