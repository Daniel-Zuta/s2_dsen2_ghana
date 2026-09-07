"""Wald's-protocol (input, target) pair construction from native-resolution patches.

`patches.py` saves patches at native resolution with no downsampling, specifically so the
degradation method used here can be changed without re-extracting patches (see agents.md
"Patch sampling strategy" and notebook 3's description). This module does that degradation
and assembles the actual training pairs for the 20m->10m network.
"""

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter, zoom
from torch.utils.data import Dataset

from . import config


def _degrade(bands: np.ndarray, factor: int) -> np.ndarray:
    """Low-pass filter + decimate a (C, H, W) array by `factor`, approximating what a
    sensor with `factor`x coarser resolution would have observed.

    Gaussian sigma is chosen so the filter's FWHM equals `factor` pixels — a standard
    heuristic for simulating resolution degradation prior to decimation (avoids the
    aliasing a naive strided downsample would introduce). This is a reasonable
    approximation, not a reproduction of Sentinel-2's actual per-band MTF (consistent
    with the clean-room reimplementation called out in agents.md's Scope decision).
    """
    sigma = factor / 2.3548  # FWHM = 2*sqrt(2*ln(2)) * sigma ≈ 2.3548 * sigma
    blurred = np.stack([gaussian_filter(band, sigma=sigma) for band in bands])
    return blurred[:, ::factor, ::factor]


def _upsample_bicubic(bands: np.ndarray, out_size: int) -> np.ndarray:
    """Upsample a (C, H, W) array so each spatial dimension equals `out_size`, via
    cubic-spline interpolation (scipy's `order=3` — the standard bicubic stand-in).
    """
    zoom_factor = out_size / bands.shape[-1]
    return np.stack([zoom(band, zoom_factor, order=3) for band in bands])


class WaldPairDataset(Dataset):
    """One sample = one native-resolution patch (see `patches.py`), converted into a
    Wald's-protocol (input, target) pair for the 20m->10m network:

    - `target`: the real `hr_20m` patch at native resolution — genuine ground truth,
      not simulated.
    - `guide`: `hr_10m` degraded by `downsample_factor` — a simulated 10m-native band at
      the same *relative* scale gap the network sees at real inference (where `guide` is
      simply the real, undegraded `hr_10m`).
    - `upsampled`: `hr_20m` degraded then bicubic-upsampled back to `hr_20m`'s native
      resolution — the network's low-res input and the residual base its prediction is
      added to (mirrors real inference, where this is the real `hr_20m` upsampled to
      `hr_10m`'s resolution).

    Returns `(guide, upsampled, target)`, each a float32 tensor, matching
    `model.DSen2Net20m.forward(guide, upsampled)`.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        downsample_factor: int = config.DOWNSAMPLE_FACTOR_20M,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.downsample_factor = downsample_factor

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        data = np.load(row["path"])
        hr_10m, hr_20m = data["hr_10m"], data["hr_20m"]

        guide = _degrade(hr_10m, self.downsample_factor)
        degraded_20m = _degrade(hr_20m, self.downsample_factor)
        upsampled = _upsample_bicubic(degraded_20m, out_size=hr_20m.shape[-1])

        return (
            torch.from_numpy(guide.astype("float32")),
            torch.from_numpy(upsampled.astype("float32")),
            torch.from_numpy(hr_20m.astype("float32")),
        )


def train_val_split_by_scene(manifest: pd.DataFrame, val_fraction: float = 0.15, seed: int = 0):
    """Split `manifest` into (train, val) by `scene_id`, not by patch — patches from the
    same scene are spatially correlated (often overlapping or adjacent), so a patch-level
    split would leak validation signal into training (see agents.md "Patch sampling
    strategy", point 4).
    """
    # `.unique()` can return a pandas extension array (e.g. StringArray) depending on how the
    # column was loaded — `rng.shuffle` isn't guaranteed correct on those (may duplicate
    # entries), which would silently break the leakage guarantee this function exists for.
    # Force a plain object-dtype ndarray first.
    scene_ids = np.asarray(manifest["scene_id"].unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(scene_ids)

    n_val = max(1, round(len(scene_ids) * val_fraction))
    val_scenes = set(scene_ids[:n_val])

    val_mask = manifest["scene_id"].isin(val_scenes)
    return manifest[~val_mask].reset_index(drop=True), manifest[val_mask].reset_index(drop=True)
