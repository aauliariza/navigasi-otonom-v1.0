"""End-to-end SUN RGB-D preparation for both tasks (proposal Sub-bab 3.2.3).

Steps:
1. Read the OFFICIAL split (``allsplit.mat``: ``alltrain`` 5,285 -> train, ``alltest`` 5,050 -> val).
2. Discover scenes and pair RGB with depth by scene identity (``pair_rgb_depth``).
3. Decode depth (Eq. 3.1-3.2), restrict to 0.001-10 m, save as uint16 mm PNG (``prepare_depth``).
4. Convert 2D boxes to single-class YOLO labels (``prepare_annotations``).
5. Write a manifest, Ultralytics dataset YAMLs, and a preparation report.

The same RGB file is used by the detection dataset (``labels/``) and the depth dataset (``depth/``):
Ultralytics maps ``images/<split>/<id>.jpg`` to ``labels/<split>/<id>.txt`` and ``depth/<split>/<id>.png``.

Usage::

    python -m src.data.prepare_sunrgbd --config configs/dataset/sunrgbd.yaml
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.pair_rgb_depth import check_pair, discover_scenes, read_intrinsics, scene_relpath
from src.data.prepare_annotations import CLASS_NAME, convert_boxes, load_annotation2d, load_meta2dbb, write_yolo_label
from src.data.prepare_depth import read_sunrgbd_depth, save_metric_depth_png
from src.utils.config import apply_overrides, load_config, resolve_path
from src.utils.io import save_csv, save_json, save_yaml
from src.utils.logger import get_logger

LOG = get_logger("prepare_sunrgbd")


def load_official_split(split_file: str | Path) -> dict[str, list[str]]:
    """Read ``allsplit.mat`` -> {"train": [scene_id...], "val": [scene_id...]}.

    A ``.json`` file ``{"train": [...], "val": [...]}`` (or ``"test"``) is also accepted. Scenes listed in both
    subsets raise an error, since that would leak validation data into training.
    """
    split_file = Path(split_file)
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Official split not found: {split_file}\n  Download https://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip "
            "and use SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat"
        )
    if split_file.suffix.lower() == ".json":
        import json

        raw = json.loads(split_file.read_text(encoding="utf-8"))
        train, val = raw["train"], raw.get("val", raw.get("test"))
    else:
        from scipy.io import loadmat

        mat = loadmat(str(split_file), squeeze_me=True, struct_as_record=False)
        if "alltrain" not in mat or "alltest" not in mat:
            raise KeyError(f"{split_file} must contain 'alltrain' and 'alltest'; found {sorted(mat)}")
        train, val = [str(x) for x in np.atleast_1d(mat["alltrain"])], [str(x) for x in np.atleast_1d(mat["alltest"])]
    split = {"train": sorted({scene_relpath(s) for s in train}), "val": sorted({scene_relpath(s) for s in val})}
    overlap = set(split["train"]) & set(split["val"])
    if overlap:
        raise ValueError(f"Official split inconsistent: {len(overlap)} scenes in both train and val")
    return split


def prepare(cfg: dict) -> dict:
    """Run the full preparation and return the report dict."""
    src, dcfg, acfg, out = cfg["source"], cfg["depth"], cfg["annotation"], cfg["output"]
    root = resolve_path(src["sunrgbd_root"])
    out_root = resolve_path(out["root"])
    split = load_official_split(resolve_path(src["split_file"]))
    LOG.info("Official split: train=%d val=%d", len(split["train"]), len(split["val"]))

    scenes = discover_scenes(root, src.get("depth_dir", "depth_bfx"))
    LOG.info("Discovered %d scenes under %s", len(scenes), root)

    if src.get("bbox_source", "meta2dbb") == "meta2dbb":
        meta = load_meta2dbb(resolve_path(src["bbox_meta"]))
        LOG.info("Loaded 2D boxes for %d scenes from %s", len(meta), src["bbox_meta"])
    else:
        meta = None

    for sub in ("images", "depth", "labels"):
        for s in split:
            d = out_root / sub / s
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    manifest, failures = [], []
    box_stats = {s: Counter() for s in split}
    counts = Counter()
    for subset, ids in split.items():
        for scene_id in ids:
            scene_dir = scenes.get(scene_id)
            if scene_dir is None:
                failures.append({"split": subset, "scene_id": scene_id, "error": "scene listed in split but not found"})
                continue
            rec = check_pair(scene_id, scene_dir, src.get("depth_dir", "depth_bfx"), cfg["validation"]["min_valid_depth_ratio"])
            if not rec.ok:
                failures.append({"split": subset, "scene_id": scene_id, "error": "; ".join(rec.errors)})
                continue
            h, w = rec.rgb_shape
            depth = read_sunrgbd_depth(rec.depth_path, dcfg["min_depth"], dcfg["max_depth"], dcfg["out_of_range"])
            raw_boxes = meta.get(scene_id, []) if meta is not None else load_annotation2d(scene_dir)
            rows, stats = convert_boxes(raw_boxes, w, h, acfg["min_box_side"])
            box_stats[subset].update(stats)

            img_out = out_root / "images" / subset / f"{rec.sample_id}{Path(rec.rgb_path).suffix.lower()}"
            shutil.copy2(rec.rgb_path, img_out)
            save_metric_depth_png(out_root / "depth" / subset / f"{rec.sample_id}.png", depth, dcfg["png_scale"])
            write_yolo_label(out_root / "labels" / subset / f"{rec.sample_id}.txt", rows)
            k = read_intrinsics(rec.intrinsics_path)
            manifest.append(
                {
                    "sample_id": rec.sample_id,
                    "scene_id": scene_id,
                    "split": subset,
                    "sensor": scene_id.split("/")[0],
                    "image": img_out.relative_to(out_root).as_posix(),
                    "height": h,
                    "width": w,
                    "num_boxes_raw": len(raw_boxes),
                    "num_boxes": len(rows),
                    "valid_depth_ratio": round(rec.valid_depth_ratio, 6),
                    "fx": k[0, 0] if k is not None else "",
                    "fy": k[1, 1] if k is not None else "",
                    "cx": k[0, 2] if k is not None else "",
                    "cy": k[1, 2] if k is not None else "",
                }
            )
            counts[subset] += 1

    meta_dir = out_root / "meta"
    save_csv(manifest, meta_dir / "manifest.csv")
    save_csv(failures, meta_dir / "preparation_failures.csv", ["split", "scene_id", "error"])

    common = {"path": str(out_root), "train": "images/train", "val": "images/val", "channels": 3}
    save_yaml({**common, "nc": 1, "names": {acfg["class_id"]: acfg.get("class_name", CLASS_NAME)}}, resolve_path(out["detect_yaml"]))
    save_yaml(
        {**common, "nc": 1, "names": {0: "depth"}, "depth_scale": dcfg["png_scale"], "min_depth": dcfg["min_depth"], "max_depth": dcfg["max_depth"]},
        resolve_path(out["depth_yaml"]),
    )
    report = {
        "split_file": str(src["split_file"]),
        "official_split_sizes": {k: len(v) for k, v in split.items()},
        "prepared_pairs": dict(counts),
        "failures": len(failures),
        "box_status": {k: dict(v) for k, v in box_stats.items()},
        "boxes_kept": {s: int(sum(m["num_boxes"] for m in manifest if m["split"] == s)) for s in split},
        "depth_source": src.get("depth_dir", "depth_bfx"),
        "bbox_source": src.get("bbox_source", "meta2dbb"),
        "depth_range_m": [dcfg["min_depth"], dcfg["max_depth"]],
        "out_of_range": dcfg["out_of_range"],
    }
    save_json(report, meta_dir / "preparation_report.json")
    LOG.info("Prepared %s, failures=%d -> %s", dict(counts), len(failures), out_root)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/dataset/sunrgbd.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="Override config values, e.g. output.root=datasets/x")
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    prepare(cfg)
    LOG.info("Next: python scripts/data/validate_dataset.py --config %s", args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
