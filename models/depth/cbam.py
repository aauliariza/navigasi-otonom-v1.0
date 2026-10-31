"""Convolutional Block Attention Module (CBAM).

Reference: S. Woo, J. Park, J.-Y. Lee, I. S. Kweon, "CBAM: Convolutional Block Attention Module", ECCV 2018,
LNCS 11211, pp. 3-19, doi:10.1007/978-3-030-01234-2_1. Authors' code: https://github.com/Jongchan/attention-module
(``MODELS/cbam.py``; see that repository for its license).

Adapted (modified): structure follows the paper/official code -- channel attention ``M_c`` from a shared
MLP (reduction 16) applied to global average- and max-pooled descriptors, then spatial attention ``M_s`` from
a 7x7 conv (+BN, as ``BasicConv`` in the official code) over the channel-wise [max, mean] maps; output
``F'' = M_s(F') * F'`` with ``F' = M_c(F) * F``. Only the constructor is changed to the Ultralytics YAML row
form ``[from, 1, CBAM, [channels, reduction, kernel_size]]``. Channel count is preserved.
"""

from __future__ import annotations

import torch
from torch import nn


class ChannelGate(nn.Module):
    """Channel attention: sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(nn.functional.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(nn.functional.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + mx)[:, :, None, None]


class SpatialGate(nn.Module):
    """Spatial attention: sigmoid(BN(Conv7x7([MaxPool_c(F); AvgPool_c(F)])))."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.amax(dim=1, keepdim=True), x.mean(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.bn(self.conv(pooled)))


class CBAM(nn.Module):
    """CBAM block: channel gate followed by spatial gate (sequential arrangement, best in the paper).

    Args:
        channels (int): Input = output channels.
        reduction (int): MLP reduction ratio (paper default 16).
        kernel_size (int): Spatial attention kernel size (paper default 7).
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction)
        self.spatial_gate = SpatialGate(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_gate(self.channel_gate(x))
