"""End-to-end SUN RGB-D preparation for both tasks (proposal Sub-bab 3.2.3).

Steps:
1. Read the OFFICIAL split (``allsplit.mat``: ``alltrain`` 5,285, ``alltest`` 5,050) and build the experiment
   split according to ``split.mode``:
   * ``proposal`` (default, proposal Tabel 3.6): ``alltrain`` -> ``train`` (5,285), ``alltest`` -> ``val`` (5,050);
     no separate test split;
   * ``internal``: ``alltest`` -> ``test`` and a scene-aware internal ``val`` carved out of ``alltrain``
     (``create_split``); ``train`` = the rest.
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

from src.data.create_split import check_group_leakage, internal_validation_split, load_group_overrides
from src.data.pair_rgb_depth import check_pair, discover_scenes, read_intrinsics, scene_relpath
from src.data.prepare_annotations import CLASS_NAME, convert_boxes, load_annotation2d, load_meta2dbb, write_yolo_label
from src.data.prepare_depth import read_sunrgbd_depth, save_metric_depth_png
from src.utils.config import apply_overrides, load_config, resolve_path
from src.utils.io import save_csv, save_json, save_yaml
from src.utils.logger import get_logger

LOG = get_logger("prepare_sunrgbd")


def load_official_split(split_file: str | Path) -> dict[str, list[str]]:
    """Read ``allsplit.mat`` -> {"train_official": [scene_id...], "test": [scene_id...]}.

    A ``.json`` file ``{"train": [...], "test": [...]}`` is also accepted (legacy key ``val`` is read as test).
    Scenes listed in both subsets raise an error, since that would leak test data into training.
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
        train, test = raw["train"], raw.get("test", raw.get("val"))
    else:
        from scipy.io import loadmat

        mat = loadmat(str(split_file), squeeze_me=True, struct_as_record=False)
        if "alltrain" not in mat or "alltest" not in mat:
            raise KeyError(f"{split_file} must contain 'alltrain' and 'alltest'; found {sorted(mat)}")
        train, test = [str(x) for x in np.atleast_1d(mat["alltrain"])], [str(x) for x in np.atleast_1d(mat["alltest"])]
    split = {"train_official": sorted({scene_relpath(s) for s in train}), "test": sorted({scene_relpath(s) for s in test})}
    overlap = set(split["train_official"]) & set(split["test"])
    if overlap:
        raise ValueError(f"Official split inconsistent: {len(overlap)} scenes in both train and test")
    return split


SPLIT_MODES = ("proposal", "internal")


def build_experiment_split(cfg: dict) -> dict:
    """Experiment split -> {"mode", "train", "val", "test", "groups", ...} (saved as ``meta/splits.json``).

    ``proposal``: train = alltrain (5,285), val = alltest (5,050), test = [] (keputusan S-1).
    ``internal``: train + val (scene-grouped) = alltrain, test = alltest (keputusan S-2).
    """
    src, scfg = cfg["source"], cfg["split"]
    mode = str(scfg.get("mode", "proposal"))
    if mode not in SPLIT_MODES:
        raise ValueError(f"split.mode must be one of {SPLIT_MODES}, got '{mode}'")
    official = load_official_split(resolve_path(src["split_file"]))
    if mode == "proposal":
        return {
            "mode": mode, "train": official["train_official"], "val": official["test"], "test": [],
            "official_sizes": {"alltrain": len(official["train_official"]), "alltest": len(official["test"])},
            "internal_val": None, "groups": {},
        }
    iv = scfg.get("internal_val", {})
    inner = internal_validation_split(
        official["train_official"], float(iv.get("fraction", 0.10)), int(iv.get("seed", 42)),
        load_group_overrides(resolve_path(iv["group_overrides"]) if iv.get("group_overrides") else None),
    )
    leaks = check_group_leakage(inner, inner["groups"])
    if leaks:
        raise ValueError(f"Internal validation shares {len(leaks)} recording groups with train: {leaks[:5]}")
    record = {
        "mode": mode, "train": inner["train"], "val": inner["val"], "test": official["test"],
        "official_sizes": {"alltrain": len(official["train_official"]), "alltest": len(official["test"])},
        "internal_val": {"fraction": float(iv.get("fraction", 0.10)), "seed": int(iv.get("seed", 42)),
                         "num_groups_train": len({inner["groups"][s] for s in inner["train"]}),
                         "num_groups_val": len({inner["groups"][s] for s in inner["val"]})},
        "groups": inner["groups"],
    }
    return record


def prepare(cfg: dict) -> dict:
    """Run the full preparation and return the report dict."""
    src, dcfg, acfg, out = cfg["source"], cfg["depth"], cfg["annotation"], cfg["output"]
    root = resolve_path(src["sunrgbd_root"])
    out_root = resolve_path(out["root"])
    split_record = build_experiment_split(cfg)
    split = {k: split_record[k] for k in ("train", "val", "test") if split_record[k]}
    LOG.info("Split mode '%s': %s", split_record["mode"], {k: len(v) for k, v in split.items()})

    scenes = discover_scenes(root, src.get("depth_dir", "depth_bfx"))
    LOG.info("Discovered %d scenes under %s", len(scenes), root)

    if src.get("bbox_source", "meta2dbb") == "meta2dbb":
        meta = load_meta2dbb(resolve_path(src["bbox_meta"]))
        LOG.info("Loaded 2D boxes for %d scenes from %s", len(meta), src["bbox_meta"])
    else:
        meta = None

    for sub in ("images", "depth", "labels"):
        for s in ("train", "val", "test"):  # also removes a stale split of the other mode
            d = out_root / sub / s
            if d.exists():
                shutil.rmtree(d)
            if s in split:
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
                    "group": split_record["groups"].get(scene_id, scene_id),
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
    save_json(split_record, meta_dir / "splits.json")

    common = {"path": str(out_root), "train": "images/train", "val": "images/val", "channels": 3}
    if "test" in split:
        common["test"] = "images/test"
    save_yaml({**common, "nc": 1, "names": {acfg["class_id"]: acfg.get("class_name", CLASS_NAME)}}, resolve_path(out["detect_yaml"]))
    save_yaml(
        {**common, "nc": 1, "names": {0: "depth"}, "depth_scale": dcfg["png_scale"], "min_depth": dcfg["min_depth"], "max_depth": dcfg["max_depth"]},
        resolve_path(out["depth_yaml"]),
    )
    report = {
        "split_file": str(src["split_file"]),
        "split_mode": split_record["mode"],
        "official_split_sizes": split_record["official_sizes"],
        "experiment_split_sizes": {k: len(v) for k, v in split.items()},
        "internal_val": split_record["internal_val"],
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
