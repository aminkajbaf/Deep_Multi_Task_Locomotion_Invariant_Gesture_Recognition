#!/usr/bin/env python3
"""Apply AIS (Algorithm 1) to Range–Doppler .npy cubes and save RAP frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from lmadgr.data import process_rd_to_rap


def main():
    parser = argparse.ArgumentParser(description="AIS preprocessing → RAP .npy")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder of RD cubes (.npy)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ais = cfg.get("ais", {})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob("*.npy"))
    print(f"Found {len(files)} files")
    for path in files:
        cube = np.load(path)
        # Support (T, M, K, L) sequences or single (M, K, L)
        if cube.ndim == 4:
            raps = [
                process_rd_to_rap(
                    cube[t],
                    doppler_cutoff=ais.get("doppler_cutoff", 2),
                    gamma=ais.get("gamma", 1.0),
                    angle_fftsize=ais.get("angle_fftsize", 64),
                )
                for t in range(cube.shape[0])
            ]
            out = np.stack(raps, axis=0)
        else:
            out = process_rd_to_rap(
                cube,
                doppler_cutoff=ais.get("doppler_cutoff", 2),
                gamma=ais.get("gamma", 1.0),
                angle_fftsize=ais.get("angle_fftsize", 64),
            )
        dest = args.output_dir / path.name
        np.save(dest, out)
        print(f"Saved {dest} shape={out.shape}")


if __name__ == "__main__":
    main()
