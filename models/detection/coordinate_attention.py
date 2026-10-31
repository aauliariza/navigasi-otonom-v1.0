"""Coordinate Attention (CA).

Reference: Q. Hou, D. Zhou, J. Feng, "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021,
pp. 13713-13722, doi:10.1109/CVPR46437.2021.01350. Official code (URL given in the paper): https://github.com/Andrew-Qibin/CoordAttention
(``coordatt.py``; see that repository for its license).

Adapted (modified) for this repository: the layer keeps the official structure (X/Y average pooling,
shared 1x1 conv + BN + h-swish with ``mip = max(8, C // reduction)``, two 1x1 convs + sigmoid producing
``a_h`` and ``a_w``, output ``x * a_w * a_h``) and only changes the constructor signature so that it can be
declared as a row of an Ultralytics model YAML (``[from, 1, CoordAtt, [channels, reduction]]``). Input and
output channel counts are identical, so the layer can be inserted in front of the Detect head.
"""

from __future__ import annotations

import torch
from torch import nn


class HSigmoid(nn.Module):
    """Hard sigmoid ``ReLU6(x + 3) / 6`` (official ``h_sigmoid``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu6(x + 3.0) / 6.0


class HSwish(nn.Module):
    """Hard swish ``x * h_sigmoid(x)`` (official ``h_swish``)."""

    def __init__(self):
        super().__init__()
        self.sigmoid = HSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate Attention block (Hou et al., 2021).

    Args:
        channels (int): Input = output channels.
        reduction (int): Channel reduction ratio of the shared transform (official default 32).
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mip = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish()
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Re-weight ``x`` (B, C, H, W) with direction-aware attention maps."""
        identity = x
        _, _, h, w = x.shape
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_h = self.conv_h(x_h).sigmoid()  # (B, C, H, 1)
        a_w = self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()  # (B, C, 1, W)
        return identity * a_w * a_h
