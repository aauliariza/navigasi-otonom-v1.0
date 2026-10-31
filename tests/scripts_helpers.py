"""Helpers to build the synthetic dataset inside a temporary folder for tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_synthetic_dataset(tmp: Path, n_train: int = 6, n_val: int = 3) -> dict:
    spec = importlib.util.spec_from_file_location("mk", ROOT / "scripts" / "data" / "make_synthetic_sunrgbd.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    raw = tmp / "raw"
    mk.main(["--out", str(raw), "--train", str(n_train), "--val", str(n_val)])
    from src.data.prepare_sunrgbd import prepare
    from src.data.validate_dataset import validate
    from src.utils.config import apply_overrides, load_config

    cfg = apply_overrides(
        load_config("configs/dataset/sunrgbd.yaml"),
        [
            f"source.sunrgbd_root={raw / 'SUNRGBD'}",
            f"source.split_file={raw / 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'}",
            f"source.bbox_meta={raw / 'SUNRGBDMeta2DBB_v2.mat'}",
            f"output.root={tmp / 'data'}",
            f"output.detect_yaml={tmp / 'data/detect.yaml'}",
            f"output.depth_yaml={tmp / 'data/depth.yaml'}",
            f"output.report_dir={tmp / 'report'}",
            f"split.expected_counts.train={n_train}",
            f"split.expected_counts.val={n_val}",
        ],
    )
    prep = prepare(cfg)
    rep = validate(cfg)
    return {"cfg": cfg, "prep": prep, "report": rep, "root": tmp / "data", "tmp": tmp, "raw": raw}
