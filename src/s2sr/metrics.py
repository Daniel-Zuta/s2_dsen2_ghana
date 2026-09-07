"""Image-quality metrics for comparing predicted vs. ground-truth super-resolved bands.

Factored out here per agents.md "Conventions for agents working here" rather than
duplicated across notebooks.
"""

import torch

# Sentinel-2 L2A surface reflectance is stored as reflectance * 10000 (standard ESA/Copernicus
# scaling for the product) — used as the nominal peak signal for PSNR unless overridden.
S2_L2A_DATA_RANGE = 10000.0


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = S2_L2A_DATA_RANGE) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB, averaged over the batch. Higher is better.

    `pred`/`target`: (B, C, H, W).
    """
    mse = torch.mean((pred - target) ** 2, dim=(-3, -2, -1))
    mse = torch.clamp(mse, min=1e-10)  # avoid log(0) for a (near-)perfect prediction
    return (10 * torch.log10(data_range**2 / mse)).mean()


def sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spectral angle mapper in degrees, averaged over all pixels and the batch. Lower is
    better — the angle between each pixel's predicted and true band-vector, independent of
    overall brightness (so it measures spectral-*shape* fidelity specifically, complementing
    PSNR which is brightness/magnitude-sensitive).

    `pred`/`target`: (B, C, H, W).
    """
    dot = (pred * target).sum(dim=-3)
    pred_norm = pred.norm(dim=-3)
    target_norm = target.norm(dim=-3)
    cos_angle = torch.clamp(dot / (pred_norm * target_norm + eps), -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_angle)).mean()
