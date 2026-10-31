"""Prediction Knowledge Distillation for the YOLO26 Detect head.

YOLO26 is anchor-free, DFL-free (``reg_max = 1``) and end-to-end: the head emits, for every anchor point ``n``
and for both the one-to-many and one-to-one branches, a class logit ``s_n`` (``nc = 1`` here) and four box
distances ``b_n = (l, t, r, b)`` in stride units. Teacher (YOLO26x) and student (YOLO26n) share strides 8/16/32
and the input size, so their predictions are anchor-aligned and can be compared one-to-one.

Two prediction terms are combined (Eq. 2.7 ``L_OD = L_det + lambda_OD * L_KD^OD``):

* Classification KD -- soft-target distillation (Hinton et al., 2015, arXiv:1503.02531), in the per-class
  sigmoid form used by YOLO classifiers: ``BCE(sigmoid(s_S / T), sigmoid(s_T / T)) * T^2``, summed over all
  anchors (foreground *and* background carry teacher knowledge, cf. Chen et al., NeurIPS 2017, "Learning
  Efficient Object Detection Models with Knowledge Distillation") and normalised by the teacher's total
  confidence ``sum_n w_n`` (mirrors Ultralytics' ``target_scores_sum`` normalisation of its own BCE).
* Box KD -- L1 between student and teacher box distances, weighted by the teacher confidence
  ``w_n = max_c sigmoid(s_T)``, normalised by ``4 * sum_n w_n``. Weighting restricts box imitation to locations
  where the teacher actually predicts an object (background box outputs are meaningless), the same weighting
  principle used by Ultralytics' feature KD (``loss_sl2``).

Status: adapted / experimental implementation (no official YOLO26 prediction-KD exists); all weights are
configurable (``kd.prediction`` in configs/detection/*.yaml).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _branch(preds, name: str) -> dict:
    if isinstance(preds, tuple):  # eval-mode head output (y, raw_dict)
        preds = preds[1]
    if isinstance(preds, dict) and name in preds:
        return preds[name]
    if isinstance(preds, dict) and "scores" in preds:
        return preds
    raise ValueError(f"Cannot find '{name}' head branch in predictions of type {type(preds)}")


def prediction_kd_branch(student: dict, teacher: dict, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Classification and box KD for one head branch.

    Args:
        student (dict): ``{"scores": (B, nc, N), "boxes": (B, 4, N)}`` raw student head output.
        teacher (dict): Same structure, from the frozen teacher.
        temperature (float): Softening temperature ``T``.

    Returns:
        (cls_kd, box_kd): Two scalar tensors.
    """
    s_cls, t_cls = student["scores"], teacher["scores"].detach().to(student["scores"].dtype)
    s_box, t_box = student["boxes"], teacher["boxes"].detach().to(student["boxes"].dtype)
    if s_cls.shape != t_cls.shape or s_box.shape != t_box.shape:
        raise ValueError(f"Teacher/student head outputs are not anchor-aligned: {s_cls.shape} vs {t_cls.shape}")
    w = t_cls.sigmoid().amax(dim=1)  # (B, N) teacher confidence
    norm = w.sum().clamp(min=1.0)
    soft_t = (t_cls / temperature).sigmoid()
    cls_kd = F.binary_cross_entropy_with_logits(s_cls / temperature, soft_t, reduction="none").sum() * temperature**2 / norm
    box_kd = ((s_box - t_box).abs().sum(dim=1) * w).sum() / (4.0 * norm)
    return cls_kd, box_kd


def prediction_kd_loss(
    student_preds,
    teacher_preds,
    temperature: float = 1.0,
    cls_weight: float = 1.0,
    box_weight: float = 1.0,
    branches: tuple[str, ...] = ("one2many", "one2one"),
) -> dict[str, torch.Tensor]:
    """Prediction KD averaged over the requested head branches.

    Returns:
        dict with ``cls`` and ``box`` (unweighted means over branches) and ``total`` (weighted sum).
    """
    cls_terms, box_terms = [], []
    for b in branches:
        c, x = prediction_kd_branch(_branch(student_preds, b), _branch(teacher_preds, b), temperature)
        cls_terms.append(c)
        box_terms.append(x)
    cls_kd, box_kd = torch.stack(cls_terms).mean(), torch.stack(box_terms).mean()
    return {"cls": cls_kd, "box": box_kd, "total": cls_weight * cls_kd + box_weight * box_kd}
