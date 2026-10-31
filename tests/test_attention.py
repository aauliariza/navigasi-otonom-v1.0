"""Coordinate Attention, CBAM and their insertion into YOLO26 / YOLO26-Depth."""

import torch

from models.common.model_utils import attention_yaml, build_from_yaml, model_stats, transfer_weights
from models.depth.cbam import CBAM
from models.detection.coordinate_attention import CoordAtt


def test_coordatt_shape_and_structure():
    m = CoordAtt(64, reduction=32)
    x = torch.randn(2, 64, 20, 12)
    y = m(x)
    assert y.shape == x.shape
    assert m.conv1.out_channels == max(8, 64 // 32)  # official mip
    assert torch.isfinite(y).all()


def test_coordatt_is_bounded_reweighting():
    m = CoordAtt(16).eval()
    x = torch.rand(1, 16, 8, 8)
    y = m(x)
    assert torch.all(y.abs() <= x.abs() + 1e-6)  # sigmoid gates in (0, 1)


def test_cbam_shape_and_gates():
    m = CBAM(32, reduction=16, kernel_size=7).eval()
    x = torch.rand(2, 32, 10, 10)
    y = m(x)
    assert y.shape == x.shape
    assert m.channel_gate.mlp[1].out_features == 2
    assert m.spatial_gate.conv.kernel_size == (7, 7)
    assert torch.all(y <= x + 1e-6)


def test_attention_yaml_inserts_before_head():
    for arch, att, task in (("yolo26n.yaml", "ca", "detect"), ("yolo26n-depth.yaml", "cbam", "depth")):
        cfg = attention_yaml(arch, att, task=task)
        rows = cfg["head"][-4:]
        assert [r[0] for r in rows[:3]] == [16, 19, 22]
        assert rows[3][0] == [23, 24, 25]
        model = build_from_yaml(cfg, task, nc=1)
        assert list(model.model[-1].f) == [23, 24, 25]
        assert type(model.model[23]).__name__ == ("CoordAtt" if att == "ca" else "CBAM")


def test_attention_cost_is_small_and_counted():
    base = model_stats(build_from_yaml("yolo26n.yaml", "detect", nc=1), 640)
    ca = model_stats(build_from_yaml(attention_yaml("yolo26n.yaml", "ca"), "detect", nc=1), 640)
    assert 0 < ca["params_total"] - base["params_total"] < 0.05 * base["params_total"]
    assert ca["gflops"] >= base["gflops"]


def test_transfer_weights_across_inserted_layers():
    torch.manual_seed(1)
    src = build_from_yaml("yolo26n.yaml", "detect", nc=1)
    dst = build_from_yaml(attention_yaml("yolo26n.yaml", "ca"), "detect", nc=1)
    transfer_weights(dst, src, verbose=False)
    assert torch.equal(dst.model[0].conv.weight, src.model[0].conv.weight)
    assert torch.equal(dst.model[26].cv2[0][0].conv.weight, src.model[23].cv2[0][0].conv.weight)  # head moved 23 -> 26
    assert torch.equal(dst.model[26].cv3[0][-1].weight, src.model[23].cv3[0][-1].weight)
