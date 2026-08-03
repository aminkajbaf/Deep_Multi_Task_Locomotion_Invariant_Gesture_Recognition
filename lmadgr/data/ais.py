"""
Adaptive Interference Suppression (AIS) — Algorithm 1 in the LMADGR paper.

Converts Range–Doppler matrices from M receive antennas into a Range–Angle
Projection (RAP) used as network input (stacked over time → 150 channels).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def adaptive_interference_suppression(
    rd_list: Sequence[np.ndarray],
    doppler_cutoff: int = 2,
    gamma: float = 1.0,
    angle_fftsize: int = 64,
) -> np.ndarray:
    """
    Args:
        rd_list: list of M Range–Doppler matrices, each (K, L) complex or real
        doppler_cutoff: δ — half-width of static-clutter zeroing around DC
        gamma: scale factor for adaptive Doppler threshold
        angle_fftsize: N_a angular FFT size

    Returns:
        RAP: (K, N_a) float32 Range–Angle Projection
    """
    if len(rd_list) == 0:
        raise ValueError("rd_list must contain at least one RD matrix")

    rds = [np.asarray(rd) for rd in rd_list]
    k, l = rds[0].shape
    mid = l // 2

    # Zero static clutter near DC Doppler
    for rd in rds:
        lo = max(0, mid - doppler_cutoff)
        hi = min(l, mid + doppler_cutoff + 1)
        rd[:, lo:hi] = 0

    stack = np.stack([np.abs(rd) for rd in rds], axis=0)  # (M, K, L)
    # E_d = Sum_range(Mean_antennas(|RD|))
    e_d = np.sum(np.mean(stack, axis=0), axis=0)  # (L,)
    t_d = float(np.mean(e_d) + gamma * np.std(e_d))

    # Angular FFT over antenna dimension → RDA (K, L, N_a)
    # Build complex antenna cube (K, L, M)
    cube = np.stack(rds, axis=-1)
    rda = np.fft.fft(cube, n=angle_fftsize, axis=-1)
    rda = np.fft.fftshift(np.abs(rda), axes=-1)

    rap = np.zeros((k, angle_fftsize), dtype=np.float64)
    for ell in range(l):
        if e_d[ell] > t_d:
            rap += rda[:, ell, :]
    return rap.astype(np.float32)


def process_rd_to_rap(
    rd_cube: np.ndarray,
    doppler_cutoff: int = 2,
    gamma: float = 1.0,
    angle_fftsize: int = 64,
) -> np.ndarray:
    """
    Convenience wrapper.

    Args:
        rd_cube: (M, K, L) or (K, L, M) Range–Doppler per antenna
    """
    arr = np.asarray(rd_cube)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D RD cube, got shape {arr.shape}")
    # Heuristic: antenna-first if first dim is small
    if arr.shape[0] <= 16 and arr.shape[0] < arr.shape[-1]:
        rd_list = [arr[m] for m in range(arr.shape[0])]
    else:
        rd_list = [arr[:, :, m] for m in range(arr.shape[-1])]
    return adaptive_interference_suppression(
        rd_list, doppler_cutoff=doppler_cutoff, gamma=gamma, angle_fftsize=angle_fftsize
    )
