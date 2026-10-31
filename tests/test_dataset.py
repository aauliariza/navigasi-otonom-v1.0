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
    p.write_text(json.dumps({"train": ["SUNRGBD/kv1/a"], "test": ["kv1/a"]}))
    with pytest.raises(ValueError):
        load_official_split(p)


def test_prepare_and_validate_synthetic(synthetic_dataset):
    """Default split mode 'proposal': alltrain -> train (6), alltest -> val (3), no test split."""
    rep, prep, root = synthetic_dataset["report"], synthetic_dataset["prep"], synthetic_dataset["root"]
    assert rep["passed"] is True and rep["split_mode"] == "proposal"
    assert prep["prepared_pairs"] == {"train": 6, "val": 3}
    assert not (root / "images/test").exists()
    assert "test" not in __import__("yaml").safe_load((synthetic_dataset["tmp"] / "data/detect.yaml").read_text())
    # same identity in images/, depth/ and labels/; class 0 only; depth in metres within range
    for split in ("train", "val"):
        stems = {p.stem for p in (root / "images" / split).glob("*.jpg")}
        assert stems == {p.stem for p in (root / "depth" / split).glob("*.png")} == {p.stem for p in (root / "labels" / split).glob("*.txt")}
        for s in stems:
            lab = read_yolo_label(root / "labels" / split / f"{s}.txt")
            assert np.all(lab[:, 0] == 0)
            d = load_metric_depth_png(root / "depth" / split / f"{s}.png")
            assert d[d > 0].min() >= 0.001 and d.max() <= 10.0
    sets = {s: {p.stem for p in (root / "images" / s).glob("*")} for s in ("train", "val")}
    assert not sets["train"] & sets["val"]
    split = json.loads((root / "meta/splits.json").read_text())
    assert split["mode"] == "proposal" and len(split["train"]) == 6 and len(split["val"]) == 3 and split["test"] == []


def test_prepare_internal_mode(tmp_path):
    """Optional split mode 'internal': alltrain -> train + grouped val, alltest -> test."""
    from scripts_helpers import build_synthetic_dataset

    d = build_synthetic_dataset(tmp_path, extra=["split.mode=internal"])
    rep, pairs, root = d["report"], d["prep"]["prepared_pairs"], d["root"]
    assert rep["passed"] is True and rep["split_mode"] == "internal" and rep["group_leakage_train_val"] == 0
    assert pairs["train"] + pairs["val"] == 6 and pairs["test"] == 3 and pairs["val"] >= 1
    sets = {s: {p.stem for p in (root / "images" / s).glob("*")} for s in ("train", "val", "test")}
    assert not sets["train"] & sets["val"] and not sets["train"] & sets["test"] and not sets["val"] & sets["test"]
    split = json.loads((root / "meta/splits.json").read_text())
    for g in {split["groups"][s] for s in split["val"]}:
        assert not any(split["groups"][s] == g for s in split["train"])


def test_proposal_mode_rejects_wrong_counts(synthetic_dataset, tmp_path):
    import copy

    from src.data.validate_dataset import validate
    from src.utils.config import apply_overrides

    cfg = apply_overrides(copy.deepcopy(synthetic_dataset["cfg"]), [f"output.report_dir={tmp_path / 'rep'}", "split.expected_counts.alltest=5050"])
    rep = validate(cfg)
    assert rep["passed"] is False and rep["split_counts_match_official"] is False


def test_scene_group_id_rules():
    from src.data.create_split import scene_group_id

    assert scene_group_id("xtion/sun3ddata/mit_32_d507/d507_2/0000103-000004205640") == "xtion/sun3ddata/mit_32_d507"
    assert scene_group_id("kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize") == \
        "kv2/kinect2data/2014-05-26_14-23-37_260595134347"
    assert scene_group_id("kv1/NYUdata/NYU0001") == "kv1/NYUdata/NYU0001"
    assert scene_group_id("kv1/NYUdata/NYU0001", {"kv1/NYUdata/NYU0001": "room_a"}) == "room_a"


def test_internal_validation_split_is_grouped_and_reproducible():
    from src.data.create_split import check_group_leakage, internal_validation_split

    ids = [f"xtion/sun3ddata/loc{k}/seq{j}/{j:07d}-0" for k in range(10) for j in range(5)]
    ids += [f"kv1/NYUdata/NYU{i:04d}" for i in range(50)]
    a = internal_validation_split(ids, 0.2, seed=42)
    b = internal_validation_split(ids, 0.2, seed=42)
    assert a["val"] == b["val"] and a["train"] == b["train"]
    assert not check_group_leakage(a, a["groups"])
    assert set(a["train"]) | set(a["val"]) == set(ids) and not set(a["train"]) & set(a["val"])
    n_val_xtion = sum(s.startswith("xtion") for s in a["val"])
    assert n_val_xtion % 5 == 0 and 5 <= n_val_xtion <= 15  # whole location groups only, ~20% per sensor
    assert 5 <= sum(s.startswith("kv1") for s in a["val"]) <= 15
    with pytest.raises(ValueError):
        internal_validation_split(ids, 1.5)


def test_validation_detects_pairing_error(synthetic_dataset, tmp_path):
    import shutil

    from src.data.validate_dataset import validate
    from src.utils.config import apply_overrides

    root = tmp_path / "data"
    shutil.copytree(synthetic_dataset["root"], root)
    next((root / "depth" / "val").glob("*.png")).unlink()
    import copy

    cfg = apply_overrides(copy.deepcopy(synthetic_dataset["cfg"]), [f"output.root={root}", f"output.report_dir={tmp_path / 'rep'}"])
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
