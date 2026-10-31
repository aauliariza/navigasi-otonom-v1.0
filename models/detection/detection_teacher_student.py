"""Teacher-student wrapper for detection KD (YOLO26x[-CA] -> YOLO26n).

``DetectionKDModel`` subclasses Ultralytics ``DistillationModel`` (ultralytics/nn/distill_model.py, AGPL-3.0,
modified by extension -- the Ultralytics source is not edited). Inherited, unchanged behaviour:

* teacher loaded from its checkpoint, ``eval()`` and ``requires_grad=False`` permanently (``_freeze_teacher``,
  ``train()`` override); the trainer additionally freezes every ``teacher_model.*`` parameter and the EMA /
  saved checkpoints drop the teacher, so ``best.pt`` contains the plain YOLO26n student only;
* forward hooks on the Detect-head inputs (student: layers 16/19/22; CA teacher: 23/24/25 = CA outputs of the
  same P3/P4/P5 levels) and on the Detect head itself;
* Feature KD = score-weighted L2 between projected student features and teacher features (``loss_sl2``),
  weight ``dis`` (proposal Sub-bab 3.2.4.2: ``dis = 6.0``). The projector is training-only.

Added: Prediction KD on the Detect head outputs (:mod:`src.losses.detection_kd`) and per-component logging.
Which terms are active is read from :attr:`DetectionKDModel.kd_config`, set by
``src/training/train_detection_kd.py`` from the experiment YAML before the trainer builds the wrapper.

Logged loss items (columns of ``results.csv``): ``box_loss``, ``cls_loss``, ``dfl_loss`` (ground truth),
``dis_loss`` (feature KD, weighted), ``pkd_loss`` (prediction KD, weighted), ``pkd_cls``, ``pkd_box`` (raw).
"""

from __future__ import annotations

import copy

import torch
from ultralytics.nn.distill_model import DistillationModel

from src.losses.detection_kd import prediction_kd_loss

DEFAULT_KD_CONFIG = {
    "feature": {"enabled": True, "weight": 6.0},
    "prediction": {"enabled": False, "weight": 1.0, "cls_weight": 1.0, "box_weight": 1.0, "temperature": 1.0, "branches": ["one2many", "one2one"]},
}


class DetectionKDModel(DistillationModel):
    """Ultralytics DistillationModel with selectable Feature KD and/or Prediction KD."""

    kd_config: dict = copy.deepcopy(DEFAULT_KD_CONFIG)

    def __init__(self, teacher_model, student_model):
        super().__init__(teacher_model=teacher_model, student_model=student_model)
        cfg = copy.deepcopy(DEFAULT_KD_CONFIG)
        for k, v in (type(self).kd_config or {}).items():
            cfg[k] = {**cfg.get(k, {}), **v}
        if not (cfg["feature"]["enabled"] or cfg["prediction"]["enabled"]):
            raise ValueError("DetectionKDModel needs at least one of kd.feature / kd.prediction enabled")
        self.kd_cfg = cfg
        self.dis = float(cfg["feature"]["weight"])

    def _zero_items(self, device) -> dict[str, torch.Tensor]:
        z = torch.zeros(1, device=device)
        return {"dis_loss": z, "pkd_loss": z.clone(), "pkd_cls": z.clone(), "pkd_box": z.clone()}

    def feature_kd(self) -> torch.Tensor:
        """Score-weighted L2 over the neck levels from the features captured by the hooks (Ultralytics logic)."""
        n = len(self.feats_idx) - 1
        head = self._teacher_feats[n]
        scores = (self.decouple_outputs(head, "one2many")["scores"] + self.decouple_outputs(head, "one2one")["scores"]) / 2
        neck = [self._teacher_feats[i] for i in range(n)]
        parts = torch.split(scores, [f.shape[-2] * f.shape[-1] for f in neck], dim=-1)
        t_scores = tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)
        loss = torch.zeros(1, device=scores.device)
        for i in range(n):
            t = self.decouple_outputs(self._teacher_feats[i])
            s = self.projector[i](self.decouple_outputs(self._student_feats[i]))
            loss = loss + self.loss_sl2(s, t, feat_idx=i, teacher_scores=t_scores)
        return loss

    def loss(self, batch, preds=None):
        """Ground-truth detection loss + enabled KD terms (each scaled by batch size like the GT loss)."""
        img = batch["img"]
        if not self.training:  # validation-during-training: GT loss only, KD columns reported as 0
            if preds is None:
                preds = self.student_model(img)
            regular, items = self.student_model.loss(batch, preds)
            items.update(self._zero_items(img.device))
            return torch.cat([regular, torch.zeros(2, device=img.device)]), items

        self._teacher_feats.clear()
        self._student_feats.clear()
        with torch.no_grad():
            self.teacher_model(img)
        preds = self.student_model(img)
        regular, items = self.student_model.loss(batch, preds)
        items.update(self._zero_items(img.device))
        bs = img.shape[0]
        feat = torch.zeros(1, device=img.device)
        pred = torch.zeros(1, device=img.device)

        if self.kd_cfg["feature"]["enabled"]:
            feat = self.feature_kd() * self.dis
            items["dis_loss"] = feat.detach()
        if self.kd_cfg["prediction"]["enabled"]:
            p = self.kd_cfg["prediction"]
            n = len(self.feats_idx) - 1
            out = prediction_kd_loss(
                preds,
                self._teacher_feats[n],
                temperature=float(p["temperature"]),
                cls_weight=float(p["cls_weight"]),
                box_weight=float(p["box_weight"]),
                branches=tuple(p.get("branches", ("one2many", "one2one"))),
            )
            pred = (out["total"] * float(p["weight"])).reshape(1)
            items["pkd_loss"] = pred.detach()
            items["pkd_cls"] = out["cls"].detach().reshape(1)
            items["pkd_box"] = out["box"].detach().reshape(1)
        return torch.cat([regular, feat * bs, pred * bs]), items
