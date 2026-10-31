"""Teacher-student wrapper for depth KD (YOLO26x-Depth[-CBAM] -> YOLO26n-Depth).

``DepthKDModel`` extends Ultralytics ``DistillationModel`` (ultralytics/nn/distill_model.py, AGPL-3.0; modified
by subclassing) so the Ultralytics trainer handles it exactly like detection KD: frozen ``eval()`` teacher with
``requires_grad=False``, teacher excluded from the optimizer/EMA/checkpoints, ``best.pt`` = plain student
``DepthModel``, resume and NaN recovery. Differences from the detection wrapper:

* tap layers are the inputs of the ``Depth`` head instead of ``Detect`` -- student 16/19/22 (proposal Sub-bab
  3.2.4.4), CBAM teacher 23/24/25 (the CBAM-refined P3/P4/P5 of the same levels);
* the projector maps TEACHER channels to STUDENT channels (proposal: "penyesuaian channel teacher ke student");
  it is training-only and stored under ``projector`` (the name the Ultralytics trainer restores on resume);
* the loss is :class:`src.losses.combined_loss.DepthKDLoss` on top of the student's ground-truth loss.

Teacher depth space (``kd.teacher_space``): the teacher runs in eval mode, where the Ultralytics Depth head
applies the teacher's log-affine calibration ``exp(a*log d + b)`` (metric output, written into ``best.pt`` by
``src/training/train_depth.py``). ``calibrated`` (default) distils that metric output; ``raw`` undoes the
calibration (``(log d - b) / a``) to distil the uncalibrated head output, matching the student's training-mode
output space.

Logged loss items: ``dlog_loss``, ``dgrad_loss`` (ground truth) and ``kd_output``, ``kd_feature``,
``kd_boundary``, ``kd_relative`` (weighted contributions; 0 when disabled).
"""

from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.modules.head import Depth

from src.losses.combined_loss import COMPONENTS, DepthKDLoss, merge_depth_kd_config
from src.losses.feature_loss import build_feature_projector


class DepthKDModel(DistillationModel):
    """DistillationModel specialised for YOLO26-Depth (see module docstring)."""

    kd_config: dict = {}

    def __init__(self, teacher_model, student_model):
        super().__init__(teacher_model=teacher_model, student_model=student_model)
        self.kd_cfg = merge_depth_kd_config(type(self).kd_config)
        self.criterion_kd = DepthKDLoss(self.kd_cfg)
        n = len(self.feats_idx) - 1
        s_ch = [self._student_feats[i].shape[1] for i in range(n)]
        t_ch = [self._teacher_feats[i].shape[1] for i in range(n)]
        self.projector = build_feature_projector(s_ch, t_ch, self.kd_cfg["feature"]["direction"]).to(
            next(student_model.parameters()).device
        )
        self._teacher_feats.clear()
        self._student_feats.clear()

    @staticmethod
    def get_distill_layers(model: nn.Module) -> list[int]:
        """Inputs of the Depth head followed by the head index, e.g. [16, 19, 22, 23] or [23, 24, 25, 26]."""
        for m in model.model:
            if isinstance(m, Depth):
                return [*list(m.f), m.i]
        raise ValueError("No Depth head found in model")

    def decouple_outputs(self, preds, branch: str = "one2one"):
        """Depth outputs are plain tensors or {'depth': tensor}; neck features are tensors."""
        if isinstance(preds, dict) and "depth" in preds:
            return preds["depth"]
        return preds

    def _zero_items(self, device) -> dict[str, torch.Tensor]:
        return {f"kd_{c}": torch.zeros(1, device=device) for c in COMPONENTS}

    def _teacher_depth(self, n: int) -> torch.Tensor:
        d = self.decouple_outputs(self._teacher_feats[n]).float()
        if self.kd_cfg.get("teacher_space", "calibrated") == "raw":
            head = self.teacher_model.model[-1]
            d = torch.exp((d.clamp_min(1e-6).log() - head.cal_b) / head.cal_a)
        return d

    def loss(self, batch, preds=None):
        """Ground-truth depth loss + enabled KD components (KD scaled by batch size like the GT loss)."""
        img = batch["img"]
        if not self.training:
            if preds is None:
                preds = self.student_model(img)
            regular, items = self.student_model.loss(batch, preds)
            items.update(self._zero_items(img.device))
            return torch.cat([regular, torch.zeros(1, device=img.device)]), items

        self._teacher_feats.clear()
        self._student_feats.clear()
        with torch.no_grad():
            self.teacher_model(img)
        preds = self.student_model(img)
        regular, items = self.student_model.loss(batch, preds)
        n = len(self.feats_idx) - 1
        student_depth = self.decouple_outputs(preds)
        kd = self.criterion_kd(
            student_depth,
            self._teacher_depth(n),
            batch["depth"],
            student_feats=[self._student_feats[i] for i in range(n)],
            teacher_feats=[self._teacher_feats[i] for i in range(n)],
            projector=self.projector,
        )
        items.update(self._zero_items(img.device))
        for c in COMPONENTS:
            if f"{c}_w" in kd:
                items[f"kd_{c}"] = kd[f"{c}_w"].detach().reshape(1)
        self._teacher_feats.clear()
        self._student_feats.clear()
        return torch.cat([regular, (kd["total"] * img.shape[0]).reshape(1)]), items
