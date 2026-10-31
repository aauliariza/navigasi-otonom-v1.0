"""Depth Output Distillation, Eq. (2.9).

``L_depth = 1/|M| * sum_{p in M} SmoothL1( log D_S(p) - log D_T(p) )``

Computed on log-depth (scale differences become additive and residuals stay comparable from 0.5 m to 10 m),
only over the valid mask ``M`` (valid ground truth AND finite positive teacher depth). Reference for the
response-based depth KD principle: Lee et al. (2025), IEEE Access 13:2763-2782, doi:10.1109/ACCESS.2024.3523912;
Smooth L1 as in Girshick (2015), Fast R-CNN, doi:10.1109/ICCV.2015.169.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over ``mask``; returns a graph-connected 0 when the mask is empty (no NaN from 0/0)."""
    m = mask.to(values.dtype)
    denom = m.sum()
    if denom <= 0:
        return (values * 0.0).sum()
    return (values * m).sum() / denom


def depth_output_loss(log_student: torch.Tensor, log_teacher: torch.Tensor, mask: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """Eq. (2.9). All tensors (B, 1, H, W); ``log_teacher`` is detached inside."""
    res = F.smooth_l1_loss(log_student, log_teacher.detach(), reduction="none", beta=beta)
    return masked_mean(res, mask)
