"""Combined depth KD objective, Eq. (2.8) / prompt section 5.2-F:

``L_total = L_ground_truth + l_output * L_output + l_feature * L_feature + l_boundary * L_boundary + l_relative * L_relative``

``L_ground_truth`` is the official YOLO26-Depth loss (Ultralytics ``DepthLoss26``: SILog + multi-scale
gradient matching) computed on the student by the student's own criterion, identical to the baseline. Each KD
component can be switched on/off and weighted from YAML (``kd.output|feature|boundary|relative``); defaults
are the weights of the proposal's full-KD configuration, Eq. (3.10): 0.5 / 0.2 / 0.1 / 0.1 (the proposal's
object-ROI term is not part of this study, as instructed). Every component is returned separately for logging.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from src.losses.boundary_loss import boundary_loss
from src.losses.depth_output_loss import depth_output_loss
from src.losses.feature_loss import feature_distillation
from src.losses.relative_depth_loss import PairSampler, relative_depth_loss

DEFAULT_DEPTH_KD = {
    "min_depth": 0.001,
    "max_depth": 10.0,
    "teacher_space": "calibrated",  # calibrated | raw  (see DepthKDModel)
    "output": {"enabled": False, "weight": 0.5, "beta": 0.1},
    "feature": {"enabled": False, "weight": 0.2, "direction": "teacher_to_student", "normalize": True},
    "boundary": {"enabled": False, "weight": 0.1, "beta": 0.1, "space": "linear"},
    "relative": {"enabled": False, "weight": 0.1, "num_pairs": 4096, "tolerance": 0.03, "margin": 0.0, "equal_pair_loss": "squared", "seed": 42},
}
COMPONENTS = ("output", "feature", "boundary", "relative")


def merge_depth_kd_config(cfg: dict | None) -> dict:
    """Overlay a user ``kd`` section on :data:`DEFAULT_DEPTH_KD`."""
    out = copy.deepcopy(DEFAULT_DEPTH_KD)
    for k, v in (cfg or {}).items():
        out[k] = {**out[k], **v} if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def kd_valid_mask(gt: torch.Tensor, teacher_depth: torch.Tensor, size, min_depth: float, max_depth: float) -> torch.Tensor:
    """Valid mask at the student output resolution: GT in [min, max] (nearest resize) AND teacher finite > 0."""
    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    if gt.shape[-2:] != tuple(size):
        gt = F.interpolate(gt.float(), size=size, mode="nearest")
    return torch.isfinite(gt) & (gt >= min_depth) & (gt <= max_depth) & torch.isfinite(teacher_depth) & (teacher_depth > 0)


class DepthKDLoss:
    """Callable computing every enabled depth KD component (raw and weighted)."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = merge_depth_kd_config(cfg)
        if not any(self.cfg[c]["enabled"] for c in COMPONENTS):
            raise ValueError("Depth KD needs at least one enabled component (output/feature/boundary/relative)")
        r = self.cfg["relative"]
        self.sampler = PairSampler(int(r["num_pairs"]), int(r.get("seed", 42)))

    def __call__(
        self,
        student_depth: torch.Tensor,
        teacher_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        student_feats: list[torch.Tensor] | None = None,
        teacher_feats: list[torch.Tensor] | None = None,
        projector=None,
    ) -> dict[str, torch.Tensor]:
        """Args are (B, 1, h, w) metric depths at the student output resolution and GT (B, H, W) in metres.

        Returns:
            dict with ``<component>`` raw losses, ``<component>_w`` weighted losses and ``total`` (weighted KD sum).
        """
        c = self.cfg
        teacher_depth = teacher_depth.detach().to(student_depth.dtype)
        if teacher_depth.shape[-2:] != student_depth.shape[-2:]:
            teacher_depth = F.interpolate(teacher_depth, size=student_depth.shape[-2:], mode="bilinear", align_corners=True)
        mask = kd_valid_mask(gt_depth, teacher_depth, student_depth.shape[-2:], c["min_depth"], c["max_depth"])
        log_s = student_depth.clamp_min(1e-6).log()
        log_t = teacher_depth.clamp_min(1e-6).log()
        out: dict[str, torch.Tensor] = {}
        if c["output"]["enabled"]:
            out["output"] = depth_output_loss(log_s, log_t, mask, float(c["output"]["beta"]))
        if c["feature"]["enabled"]:
            if student_feats is None or teacher_feats is None or projector is None:
                raise RuntimeError("Feature KD enabled but features/projector were not provided")
            out["feature"], _ = feature_distillation(student_feats, teacher_feats, projector, c["feature"]["direction"], c["feature"]["normalize"])
        if c["boundary"]["enabled"]:
            out["boundary"] = boundary_loss(student_depth, teacher_depth, mask, float(c["boundary"]["beta"]), c["boundary"]["space"])
        if c["relative"]["enabled"]:
            r = c["relative"]
            out["relative"] = relative_depth_loss(log_s, log_t, mask, self.sampler, float(r["tolerance"]), float(r["margin"]), r["equal_pair_loss"])
        total = torch.zeros((), device=student_depth.device, dtype=student_depth.dtype)
        for k in list(out):
            out[f"{k}_w"] = out[k] * float(c[k]["weight"])
            total = total + out[f"{k}_w"]
        out["total"] = total
        return out
