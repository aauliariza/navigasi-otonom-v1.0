"""Boundary Distillation, Eq. (2.11)-(2.12).

``G_x = dD/dx, G_y = dD/dy, G = sqrt(G_x^2 + G_y^2)``;  ``L_boundary = SmoothL1(G_S, G_T)``

Gradients are forward differences; a gradient is valid only if BOTH pixels it uses are valid, so an invalid
(missing) depth pixel never creates a fake edge. ``space='linear'`` differentiates metric depth exactly as in
Eq. (2.11); ``space='log'`` differentiates log-depth (scale-invariant edges, option). Gradient-based edge
supervision for depth: Liu & Wang (2024), Electronics 13(9):1652, doi:10.3390/electronics13091652.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses.depth_output_loss import masked_mean


def gradient_magnitude(depth: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (G, valid) of shape (B, 1, H-1, W-1) from (B, 1, H, W) depth and its boolean valid mask."""
    gx = depth[..., :-1, 1:] - depth[..., :-1, :-1]
    gy = depth[..., 1:, :-1] - depth[..., :-1, :-1]
    valid = mask[..., :-1, :-1] & mask[..., :-1, 1:] & mask[..., 1:, :-1]
    return torch.sqrt(gx.pow(2) + gy.pow(2) + eps), valid


def boundary_loss(
    depth_student: torch.Tensor,
    depth_teacher: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.1,
    space: str = "linear",
) -> torch.Tensor:
    """Eq. (2.12) over valid gradient locations. Inputs are metric depth maps (B, 1, H, W)."""
    if space == "log":
        depth_student, depth_teacher = depth_student.clamp_min(1e-6).log(), depth_teacher.clamp_min(1e-6).log()
    elif space != "linear":
        raise ValueError(f"space must be 'linear' or 'log', got '{space}'")
    gs, valid = gradient_magnitude(depth_student, mask)
    gt, _ = gradient_magnitude(depth_teacher.detach(), mask)
    return masked_mean(F.smooth_l1_loss(gs, gt, reduction="none", beta=beta), valid)
