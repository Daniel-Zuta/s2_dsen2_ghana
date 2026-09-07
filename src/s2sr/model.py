"""PyTorch port of DSen2's 20m->10m super-resolution network (VDSR-style residual CNN).

Clean-room reimplementation of the published architecture (see agents.md "Framework
decision" / "Scope decision") — not a weight-for-weight port of the original Keras model.
"""

import torch
import torch.nn as nn


class DSen2Net20m(nn.Module):
    """Predicts a residual correction added to a bicubic-upsampled 20m input, guided by
    the native-resolution 10m bands.

    `forward` takes `guide` (B, guide_channels, H, W) — native-resolution 10m bands —
    and `upsampled` (B, target_channels, H, W) — the 20m bands already bicubic-upsampled
    to the same (H, W) — and returns the corrected 20m bands at (B, target_channels, H, W).
    Both inputs must already be spatially aligned and at the same resolution; see
    `s2sr.dataset.WaldPairDataset`, which builds exactly this pair (at training scale) or
    real full-resolution bands (at inference).
    """

    def __init__(
        self,
        guide_channels: int = 4,
        target_channels: int = 6,
        num_filters: int = 128,
        num_layers: int = 6,
    ):
        super().__init__()
        in_channels = guide_channels + target_channels

        layers = [nn.Conv2d(in_channels, num_filters, 3, padding=1), nn.PReLU(num_filters)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(num_filters, num_filters, 3, padding=1), nn.PReLU(num_filters)]
        layers += [nn.Conv2d(num_filters, target_channels, 3, padding=1)]

        self.body = nn.Sequential(*layers)

    def forward(self, guide: torch.Tensor, upsampled: torch.Tensor) -> torch.Tensor:
        x = torch.cat([guide, upsampled], dim=1)
        return upsampled + self.body(x)
