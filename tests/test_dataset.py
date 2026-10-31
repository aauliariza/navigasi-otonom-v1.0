"""Dataset pipeline: depth decoding, valid mask, pairing, YOLO annotation, integrity validation."""

import json

import numpy as np
import pytest

from src.data.pair_rgb_depth import sample_id_from_scene, scene_relpath
from src.data.prepare_annotations import clean_box, convert_boxes, read_yolo_label, xyxy_to_yolo, yolo_to_xyxy
from src.data.prepare_depth import (
    apply_valid_range,
    decode_sunrgbd_raw,
    depth_mm_to_m,
    encode_sunrgbd_raw,
    load_metric_depth_png,
    save_metric_depth_png,
    valid_depth_mask,
)


def test_bit_rotation_decode_matches_formula():
    raw = np.array([[8, 1, 0, 65535]], dtype=np.uint16)
    out = decode_sunrgbd_raw(raw)
    expected = [(int(v) >> 3 | int(v) << 13) & 0xFFFF for v in raw.ravel()]
    assert out.ravel().tolist() == expected
    assert out[0, 0] == 1  # 8 >> 3


def test_encode_decode_roundtrip():
    mm = np.random.default_rng(0).integers(0, 10001, (32, 32)).astype(np.uint16)
    assert np.array_equal(decode_sunrgbd_raw(encode_sunrgbd_raw(mm)), mm)


def test_decode_rejects_8bit():
    with pytest.raises(ValueError):
        decode_sunrgbd_raw(np.zeros((2, 2), np.uint8))


def test_mm_to_m_and_valid_range():
    d = depth_mm_to_m(np.array([0, 1, 1000, 10000, 12000], np.uint16))
    assert np.allclose(d, [0, 0.001, 1.0, 10.0, 12.0])
    inv = apply_valid_range(d, out_of_range="invalidate")
    assert inv.tolist() == pytest.approx([0, 0.001, 1.0, 10.0, 0.0])
    clip = apply_valid_range(d, out_of_range="clip")
    assert clip[-1] == 10.0
    assert valid_depth_mask(inv).tolist() == [False, True, True, True, False]
    with pytest.raises(ValueError):
        apply_valid_range(d, out_of_range="x")


def test_depth_png_lossless(tmp_path):
    d = np.array([[0.0, 0.001, 1.234, 9.999, np.nan]], np.float32)
    p = save_metric_depth_png(tmp_path / "d.png", d)
    back = load_metric_depth_png(p)
    assert back.dtype == np.float32
    assert np.allclose(back, [[0.0, 0.001, 1.234, 9.999, 0.0]], atol=5e-4)


def test_scene_relpath_variants():
    assert scene_relpath("/n/fs/sun3d/data/SUNRGBD/kv1/NYUdata/NYU0001/") == "kv1/NYUdata/NYU0001"
    assert scene_relpath("SUNRGBD/kv2/kinect2data/000002") == "kv2/kinect2data/000002"
    assert scene_relpath("/x/y/realsense/lg/2014_10_24-11_52_23-1311000073") == "realsense/lg/2014_10_24-11_52_23-1311000073"
    assert sample_id_from_scene("kv1/NYUdata/NYU0001") == "kv1__NYUdata__NYU0001"


def test_clean_box_rules():
    assert clean_box([10, 10, 20, 30], 100, 100) == ([10, 10, 30, 40], "ok")
    assert clean_box([-5, 10, 20, 30], 100, 100) == ([0, 10, 15, 40], "clipped")
    assert clean_box([200, 10, 20, 30], 100, 100) == (None, "outside")
    assert clean_box([10, 10, 3, 30], 100, 100) == (None, "too_small")
    assert clean_box([98, 10, 10, 30], 100, 100)[1] == "too_small"  # clipped width 2 px
    assert clean_box([10, 10, np.nan, 30], 100, 100) == (None, "invalid")
    assert clean_box([10, 10, -4, 30], 100, 100) == (None, "invalid")


def test_yolo_conversion_equations():
    # Eq. 3.3-3.8: x=10,y=20,w=40,h=60 on 200x100 image
    xc, yc, w, h = xyxy_to_yolo([10, 20, 50, 80], 200, 100)
    assert (xc, yc, w, h) == pytest.approx((30 / 200, 50 / 100, 40 / 200, 60 / 100))
    rows = np.array([[0, xc, yc, w, h]])
    assert yolo_to_xyxy(rows, 200, 100)[0] == pytest.approx([10, 20, 50, 80])
    yolo, stats = convert_boxes([[10, 20, 40, 60], [0, 0, 2, 2]], 200, 100)
    assert len(yolo) == 1 and stats["too_small"] == 1


def test_official_split_rejects_overlap(tmp_path):
    from src.data.prepare_sunrgbd import load_official_split

    p = tmp_path / "s.json"
    p.write_text(json.dumps({"train": ["SUNRGBD/kv1/a"], "val": ["kv1/a"]}))
    with pytest.raises(ValueError):
        load_official_split(p)


def test_prepare_and_validate_synthetic(synthetic_dataset):
    rep, prep, root = synthetic_dataset["report"], synthetic_dataset["prep"], synthetic_dataset["root"]
    assert rep["passed"] is True
    assert prep["prepared_pairs"] == {"train": 6, "val": 3}
    assert rep["summary"]["train"]["valid_pairs"] == 6 and rep["summary"]["val"]["valid_pairs"] == 3
    # same identity in images/, depth/ and labels/; class 0 only; depth in metres within range
    for split in ("train", "val"):
        stems = {p.stem for p in (root / "images" / split).glob("*.jpg")}
        assert stems == {p.stem for p in (root / "depth" / split).glob("*.png")} == {p.stem for p in (root / "labels" / split).glob("*.txt")}
        for s in stems:
            lab = read_yolo_label(root / "labels" / split / f"{s}.txt")
            assert np.all(lab[:, 0] == 0)
            d = load_metric_depth_png(root / "depth" / split / f"{s}.png")
            assert d[d > 0].min() >= 0.001 and d.max() <= 10.0
    assert not set(p.stem for p in (root / "images/train").glob("*")) & set(p.stem for p in (root / "images/val").glob("*"))


def test_validation_detects_pairing_error(synthetic_dataset, tmp_path):
    import shutil

    from src.data.validate_dataset import validate
    from src.utils.config import apply_overrides

    root = tmp_path / "data"
    shutil.copytree(synthetic_dataset["root"], root)
    next((root / "depth" / "val").glob("*.png")).unlink()
    cfg = apply_overrides(dict(synthetic_dataset["cfg"]), [f"output.root={root}", f"output.report_dir={tmp_path / 'rep'}"])
    rep = validate(cfg)
    assert rep["passed"] is False
    assert rep["issues_by_type"]["pairing"] >= 1


def test_integrity_gate(tmp_path):
    from src.data.validate_dataset import require_dataset_ok

    with pytest.raises(RuntimeError):
        require_dataset_ok(tmp_path / "missing.json")
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"passed": False}))
    with pytest.raises(RuntimeError):
        require_dataset_ok(p)
    p.write_text(json.dumps({"passed": True}))
    assert require_dataset_ok(p)["passed"]
