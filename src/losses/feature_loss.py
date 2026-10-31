"""Feature distillation losses.

1. :func:`score_weighted_l2` -- detection Feature KD. Identical to Ultralytics ``DistillationModel.loss_sl2``
   (ultralytics/nn/distill_model.py, AGPL-3.0): per-location MSE between the projected student feature and the
   teacher feature, weighted by the teacher's max class probability at that location, normalised by
   ``sum(weight) * C``. Re-implemented here as a pure function so it can be unit-tested and logged per level;
   :class:`models.detection.detection_teacher_student.DetectionKDModel` uses the Ultralytics method itself, and
   the unit test asserts both agree.

2. :class:`FeatureDistillationLoss` -- depth Feature Distillation, Eq. (2.10):
   ``L_feature = sum_l d(P_l(F_T^l), F_S^l)`` with a learned 1x1 projection ``P_l`` that maps the TEACHER channels
   to the STUDENT channels (proposal Sub-bab 3.2.4.4, "penyesuaian channel teacher ke student") and ``d`` = MSE
   between channel-L2-normalised maps (normalisation keeps the scale of the 3 levels comparable and prevents the
   trivial solution of shrinking both maps). The opposite direction (student -> teacher, FitNets-style) is
   available via ``direction: student_to_teacher``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def score_weighted_l2(student_feat: torch.Tensor, teacher_feat: torch.Tensor, teacher_score: torch.Tensor) -> torch.Tensor:
    """Ultralytics score-weighted L2 for one level.

    Args:
        student_feat (Tensor): Projected student feature (N, C, H, W).
        teacher_feat (Tensor): Teacher feature (N, C, H, W).
        teacher_score (Tensor): Teacher max class probability per location (N, 1, H*W).
    """
    n, c = student_feat.shape[:2]
    mse = F.mse_loss(student_feat.view(n, c, -1), teacher_feat.view(n, c, -1), reduction="none")
    return (mse * teacher_score).sum() / (teacher_score.sum() * c + 1e-9)


def build_projector(c_in: int, c_out: int) -> nn.Sequential:
    """Two-layer 1x1 conv projector (same design as Ultralytics' detection projector)."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 1), nn.ReLU(inplace=True), nn.Conv2d(c_out, c_out, 1))


class FeatureDistillationLoss(nn.Module):
    """Multi-level feature distillation with learned channel alignment (Eq. 2.10).

    Args:
        student_channels (list[int]): Channels of the student tap layers (e.g. 16/19/22).
        teacher_channels (list[int]): Channels of the teacher tap layers.
        direction (str): ``teacher_to_student`` (proposal) or ``student_to_teacher``.
        normalize (bool): L2-normalise every location's channel vector before the MSE.
    """

    def __init__(self, student_channels, teacher_channels, direction: str = "teacher_to_student", normalize: bool = True):
        super().__init__()
        if len(student_channels) != len(teacher_channels):
            raise ValueError("student and teacher must expose the same number of feature levels")
        if direction not in {"teacher_to_student", "student_to_teacher"}:
            raise ValueError(f"unknown direction '{direction}'")
        self.direction = direction
        self.normalize = normalize
        self.projector = build_feature_projector(student_channels, teacher_channels, direction)

    def forward(self, student_feats: list[torch.Tensor], teacher_feats: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return (sum over levels, per-level losses)."""
        return feature_distillation(student_feats, teacher_feats, self.projector, self.direction, self.normalize)


def build_feature_projector(student_channels, teacher_channels, direction: str = "teacher_to_student") -> nn.ModuleList:
    """Projector bank for :func:`feature_distillation` (one 2-layer 1x1 conv per level)."""
    pairs = zip(teacher_channels, student_channels) if direction == "teacher_to_student" else zip(student_channels, teacher_channels)
    return nn.ModuleList(build_projector(ci, co) for ci, co in pairs)


def feature_distillation(student_feats, teacher_feats, projector, direction: str = "teacher_to_student", normalize: bool = True):
    """Eq. (2.10): ``sum_l MSE(norm(P_l(F_T^l)), norm(F_S^l))``. Teacher features are detached (frozen teacher).

    Returns:
        (total, per_level): scalar sum over levels and the list of per-level losses.
    """
    per_level = []
    for proj, fs, ft in zip(projector, student_feats, teacher_feats):
        ft = ft.detach().to(fs.dtype)
        if ft.shape[-2:] != fs.shape[-2:]:
            ft = F.interpolate(ft, size=fs.shape[-2:], mode="bilinear", align_corners=False)
        a, b = (proj(ft), fs) if direction == "teacher_to_student" else (proj(fs), ft)
        if normalize:
            a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
        per_level.append(F.mse_loss(a, b))
    return torch.stack(per_level).sum(), per_level
