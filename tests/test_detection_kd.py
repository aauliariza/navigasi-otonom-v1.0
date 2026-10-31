"""Detection KD: feature (score-weighted L2) and prediction distillation, teacher freezing, gradient flow."""

import pytest
import torch
from ultralytics.nn.distill_model import DistillationModel

from models.detection.detection_teacher_student import DetectionKDModel
from src.losses.detection_kd import prediction_kd_branch, prediction_kd_loss
from src.losses.feature_loss import score_weighted_l2


def _batch(n=2, s=64):
    return {
        "img": torch.rand(n, 3, s, s),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.zeros(2, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.3, 0.3, 0.2, 0.3]]),
    }


def test_score_weighted_l2_matches_ultralytics():
    s, t = torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 4)
    w = torch.rand(2, 1, 16)
    ref = DistillationModel.loss_sl2(None, s, t, 0, (w,))
    assert torch.allclose(score_weighted_l2(s, t, w), ref)
    assert score_weighted_l2(t, t, w).item() == pytest.approx(0.0)


def test_prediction_kd_branch_zero_for_identical_and_positive_otherwise():
    t = {"scores": torch.randn(2, 1, 50), "boxes": torch.rand(2, 4, 50)}
    cls_same, box_same = prediction_kd_branch({k: v.clone() for k, v in t.items()}, t)
    s = {"scores": torch.randn(2, 1, 50), "boxes": torch.rand(2, 4, 50)}
    cls_diff, box_diff = prediction_kd_branch(s, t)
    # BCE against soft targets is minimal (= target entropy) at identity; box L1 exactly 0
    assert box_same.item() == pytest.approx(0.0)
    assert cls_diff.item() > cls_same.item()
    assert box_diff.item() > 0


def test_prediction_kd_gradient_only_to_student():
    t = {"scores": torch.randn(1, 1, 30, requires_grad=True), "boxes": torch.rand(1, 4, 30, requires_grad=True)}
    s = {"scores": torch.randn(1, 1, 30, requires_grad=True), "boxes": torch.rand(1, 4, 30, requires_grad=True)}
    out = prediction_kd_loss({"one2many": s, "one2one": s}, {"one2many": t, "one2one": t})
    out["total"].backward()
    assert s["scores"].grad is not None and s["boxes"].grad is not None
    assert t["scores"].grad is None and t["boxes"].grad is None


def test_prediction_kd_shape_mismatch_raises():
    with pytest.raises(ValueError):
        prediction_kd_branch({"scores": torch.zeros(1, 1, 10), "boxes": torch.zeros(1, 4, 10)},
                             {"scores": torch.zeros(1, 1, 12), "boxes": torch.zeros(1, 4, 12)})


@pytest.mark.parametrize("feature,prediction", [(True, False), (False, True), (True, True)])
def test_detection_kd_model_end_to_end(det_pair, feature, prediction):
    teacher, student = det_pair
    DetectionKDModel.kd_config = {"feature": {"enabled": feature, "weight": 6.0}, "prediction": {"enabled": prediction}}
    m = DetectionKDModel(teacher_model=teacher, student_model=student)
    m.train()
    assert m.feats_idx == [16, 19, 22, 23] and m.teacher_feats_idx == [23, 24, 25, 26]  # CA teacher taps
    assert not m.teacher_model.training
    assert all(not p.requires_grad for p in m.teacher_model.parameters())
    loss, items = m.loss(_batch())
    assert torch.isfinite(loss).all() and loss.shape[0] == 5
    assert (items["dis_loss"].item() > 0) == feature
    assert (items["pkd_loss"].item() > 0) == prediction
    loss.sum().backward()
    assert all(p.grad is None for p in m.teacher_model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.student_model.parameters())
    assert (m.projector[0][0].weight.grad is not None) == feature
    m.eval()
    vloss, vitems = m.loss(_batch())
    assert vitems["dis_loss"].item() == 0 and vitems["pkd_loss"].item() == 0


def test_detection_kd_requires_a_term(det_pair):
    DetectionKDModel.kd_config = {"feature": {"enabled": False}, "prediction": {"enabled": False}}
    with pytest.raises(ValueError):
        DetectionKDModel(teacher_model=det_pair[0], student_model=det_pair[1])
