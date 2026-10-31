"""Depth KD components, combined loss, DepthKDModel and log-affine calibration."""

import numpy as np
import pytest
import torch

from models.depth.depth_teacher_student import DepthKDModel
from models.depth.yolo26_depth import fit_log_affine
from src.losses.boundary_loss import boundary_loss, gradient_magnitude
from src.losses.combined_loss import DepthKDLoss, kd_valid_mask
from src.losses.depth_output_loss import depth_output_loss, masked_mean
from src.losses.feature_loss import FeatureDistillationLoss
from src.losses.relative_depth_loss import PairSampler, ordinal_labels, relative_depth_loss


def _depth(seed=0, shape=(2, 1, 16, 16)):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g) * 5 + 0.5


def test_masked_mean_empty_mask_is_zero_and_connected():
    v = torch.ones(1, 1, 4, 4, requires_grad=True)
    out = masked_mean(v, torch.zeros_like(v, dtype=torch.bool))
    assert out.item() == 0.0
    out.backward()


def test_output_loss_zero_identical_and_mask_respected():
    d = _depth()
    m = torch.ones_like(d, dtype=torch.bool)
    assert depth_output_loss(d.log(), d.log(), m).item() == pytest.approx(0.0)
    t = d.clone()
    t[..., :8] *= 3.0  # disagreement only in the masked-out half
    m[..., :8] = False
    assert depth_output_loss(d.log(), t.log(), m).item() == pytest.approx(0.0)
    m[...] = True
    assert depth_output_loss(d.log(), t.log(), m).item() > 0


def test_output_loss_is_log_space_scale_symmetric():
    d = _depth()
    m = torch.ones_like(d, dtype=torch.bool)
    a = depth_output_loss((2 * d).log(), d.log(), m)
    b = depth_output_loss((0.5 * d).log(), d.log(), m)
    assert a.item() == pytest.approx(b.item(), rel=1e-5)


def test_gradient_magnitude_and_boundary():
    d = torch.ones(1, 1, 6, 6)
    d[..., 3:] = 3.0  # vertical step edge
    m = torch.ones_like(d, dtype=torch.bool)
    g, valid = gradient_magnitude(d, m)
    assert g.max().item() == pytest.approx(2.0, abs=1e-4)
    assert boundary_loss(d, d, m).item() == pytest.approx(0.0, abs=1e-6)
    assert boundary_loss(torch.ones_like(d), d, m).item() > 0
    m2 = m.clone()
    m2[..., 2:4] = False  # invalidate the columns forming the edge
    assert boundary_loss(torch.ones_like(d), d, m2).item() == pytest.approx(0.0, abs=1e-6)


def test_ordinal_labels_tolerance():
    r = ordinal_labels(torch.tensor([0.5, -0.5, 0.02, -0.03, 0.031]), 0.03)
    assert r.tolist() == [1, -1, 0, 0, 1]


def test_relative_loss_prefers_correct_order():
    t = _depth(1)
    m = torch.ones_like(t, dtype=torch.bool)
    good = relative_depth_loss(t.log(), t.log(), m, PairSampler(4096, 0))
    bad = relative_depth_loss((1.0 / t).log(), t.log(), m, PairSampler(4096, 0))
    assert bad.item() > good.item()


def test_pair_sampler_reproducible():
    idx = torch.arange(1000)
    a = PairSampler(64, seed=7)(idx)
    b = PairSampler(64, seed=7)(idx)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_relative_loss_ignores_invalid_pixels():
    t = _depth(2)
    m = torch.zeros_like(t, dtype=torch.bool)
    out = relative_depth_loss(t.log(), t.log(), m, PairSampler(16))
    assert out.item() == 0.0


def test_feature_loss_projection_direction_and_grad():
    loss = FeatureDistillationLoss([8, 16], [24, 48], direction="teacher_to_student")
    s = [torch.randn(1, 8, 8, 8, requires_grad=True), torch.randn(1, 16, 4, 4, requires_grad=True)]
    t = [torch.randn(1, 24, 8, 8, requires_grad=True), torch.randn(1, 48, 4, 4, requires_grad=True)]
    total, per = loss(s, t)
    assert loss.projector[0][0].in_channels == 24 and loss.projector[0][-1].out_channels == 8
    total.backward()
    assert s[0].grad is not None and t[0].grad is None and len(per) == 2


def test_kd_valid_mask_range():
    gt = torch.tensor([[[0.0, 0.0005, 0.5, 10.0, 12.0, float("nan")]]])
    teacher = torch.ones(1, 1, 1, 6)
    m = kd_valid_mask(gt, teacher, (1, 6), 0.001, 10.0)
    assert m.flatten().tolist() == [False, False, True, True, False, False]


def test_combined_loss_weights_and_toggles():
    s, t = _depth(0), _depth(1)
    gt = _depth(2).squeeze(1)
    kd = DepthKDLoss({"output": {"enabled": True, "weight": 0.5}, "boundary": {"enabled": True, "weight": 0.1}})
    out = kd(s, t, gt)
    assert set(out) == {"output", "output_w", "boundary", "boundary_w", "total"}
    assert out["output_w"].item() == pytest.approx(0.5 * out["output"].item())
    assert out["total"].item() == pytest.approx(out["output_w"].item() + out["boundary_w"].item())
    with pytest.raises(ValueError):
        DepthKDLoss({})


@pytest.mark.parametrize("components", [("output",), ("feature",), ("boundary",), ("relative",), ("output", "feature", "boundary", "relative")])
def test_depth_kd_model_end_to_end(depth_pair, components):
    teacher, student = depth_pair
    DepthKDModel.kd_config = {c: {"enabled": True} for c in components}
    m = DepthKDModel(teacher_model=teacher, student_model=student)
    m.train()
    assert m.feats_idx == [16, 19, 22, 23] and m.teacher_feats_idx == [23, 24, 25, 26]
    assert all(not p.requires_grad for p in m.teacher_model.parameters()) and not m.teacher_model.training
    # teacher -> student projector: in = teacher channels, out = student channels
    assert m.projector[0][0].in_channels == teacher.model[23].channel_gate.mlp[1].in_features
    batch = {"img": torch.rand(2, 3, 64, 64), "depth": torch.rand(2, 64, 64) * 9 + 0.5}
    batch["depth"][:, :10] = 0.0  # invalid rows
    loss, items = m.loss(batch)
    assert torch.isfinite(loss).all()
    for c in ("output", "feature", "boundary", "relative"):
        assert (items[f"kd_{c}"].item() != 0) == (c in components)
    loss.sum().backward()
    assert all(p.grad is None for p in m.teacher_model.parameters())
    assert any(p.grad is not None for p in m.student_model.parameters())
    import copy
    import pickle

    pickle.dumps(copy.deepcopy(m))  # EMA/checkpoint path must stay picklable


def test_fit_log_affine_recovers_parameters_and_fallback():
    rng = np.random.default_rng(0)
    lp = rng.normal(0.5, 0.6, 5000)
    a, b, mode = fit_log_affine(lp, 1.3 * lp - 0.4)
    assert (a, b, mode) == (pytest.approx(1.3), pytest.approx(-0.4), "log_affine")
    a, b, mode = fit_log_affine(lp, -lp + 0.2)
    assert mode == "scale_only_fallback" and a == 1.0
