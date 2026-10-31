"""Integrity validation of the prepared SUN RGB-D dataset (proposal Sub-bab 3.2.3, prompt section 6.5).

Checks performed on ``datasets/sunrgbd_yolo``:
* number of RGB images, depth maps and label files per split; images without depth / depth without image
  / labels without image (pairing errors);
* RGB and depth resolution identical for every pair; depth is uint16 single-channel and decodes to metres;
* no NaN/Inf, valid-depth ratio per image, depth values inside 0.001-10 m;
* YOLO labels: 5 columns, class 0, normalised coordinates in [0, 1], boxes inside the image, sides >= 4 px;
* no sample identity shared by any two of train / val / test, no recording group (``meta/manifest.csv``
  ``group`` column) shared by train and val, and split sizes equal to the official partition:
  ``train + val = 5,285`` (official alltrain) and ``test = 5,050`` (official alltest).

Outputs (``outputs/dataset``): ``integrity_report.json`` (with ``passed``), ``integrity_issues.csv``,
``objects_per_image.csv``, ``objects_per_image.png`` and annotated RGB|depth sample panels. Training scripts
refuse to start until ``integrity_report.json`` exists with ``passed: true`` (see :func:`require_dataset_ok`).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.data.prepare_annotations import read_yolo_label, yolo_to_xyxy
from src.data.prepare_depth import load_metric_depth_png
from src.utils.config import apply_overrides, load_config, resolve_path
from src.utils.io import load_json, save_csv, save_json
from src.utils.logger import get_logger
from src.utils.visualization import colorize_depth, draw_boxes, side_by_side

LOG = get_logger("validate_dataset")
IMG_EXT = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")


def _stems(folder: Path, exts: set[str]) -> dict[str, Path]:
    return {p.stem: p for p in sorted(folder.glob("*")) if p.suffix.lower() in exts} if folder.is_dir() else {}


def validate(cfg: dict) -> dict:
    """Run every check and return the report (also written to disk)."""
    root = resolve_path(cfg["output"]["root"])
    report_dir = resolve_path(cfg["output"]["report_dir"])
    dmin, dmax = cfg["depth"]["min_depth"], cfg["depth"]["max_depth"]
    min_side = cfg["annotation"]["min_box_side"]
    scale = cfg["depth"]["png_scale"]
    expected = cfg["split"].get("expected_counts", {})
    issues: list[dict] = []
    per_image: list[dict] = []
    summary: dict = {}

    def issue(split, sid, kind, msg):
        issues.append({"split": split, "sample_id": sid, "type": kind, "detail": msg})

    ids_by_split = {}
    for split in SPLITS:
        imgs = _stems(root / "images" / split, IMG_EXT)
        deps = _stems(root / "depth" / split, {".png"})
        labs = _stems(root / "labels" / split, {".txt"})
        ids_by_split[split] = set(imgs)
        for sid in sorted(set(imgs) - set(deps)):
            issue(split, sid, "pairing", "RGB without depth")
        for sid in sorted(set(deps) - set(imgs)):
            issue(split, sid, "pairing", "depth without RGB")
        for sid in sorted(set(labs) - set(imgs)):
            issue(split, sid, "pairing", "label without RGB")
        for sid in sorted(set(imgs) - set(labs)):
            issue(split, sid, "pairing", "RGB without label file")

        n_valid_pairs, n_boxes, invalid_data = 0, 0, 0
        valid_ratios = []
        for sid in sorted(set(imgs) & set(deps)):
            im = cv2.imread(str(imgs[sid]), cv2.IMREAD_COLOR)
            if im is None:
                issue(split, sid, "corrupt", "unreadable RGB")
                invalid_data += 1
                continue
            try:
                d = load_metric_depth_png(deps[sid], scale)
            except OSError as e:
                issue(split, sid, "corrupt", str(e))
                invalid_data += 1
                continue
            h, w = im.shape[:2]
            bad = False
            if d.shape != (h, w):
                issue(split, sid, "resolution", f"rgb={h}x{w} depth={d.shape[0]}x{d.shape[1]}")
                bad = True
            if not np.isfinite(d).all():
                issue(split, sid, "numeric", "non-finite depth")
                bad = True
            vals = d[d > 0]
            if vals.size and (vals.min() < dmin - 1e-6 or vals.max() > dmax + 1e-6):
                issue(split, sid, "range", f"depth outside [{dmin},{dmax}] m: {vals.min():.4f}-{vals.max():.4f}")
                bad = True
            ratio = float(vals.size / d.size)
            if ratio <= cfg["validation"]["min_valid_depth_ratio"]:
                issue(split, sid, "depth", f"valid depth ratio {ratio:.4f}")
                bad = True
            valid_ratios.append(ratio)

            rows = read_yolo_label(labs[sid]) if sid in labs else np.zeros((0, 5))
            if rows.size:
                if rows.shape[1] != 5:
                    issue(split, sid, "label", f"expected 5 columns, got {rows.shape[1]}")
                    bad = True
                elif not np.all(rows[:, 0] == 0):
                    issue(split, sid, "label", "class id != 0")
                    bad = True
                elif np.any(rows[:, 1:] < 0) or np.any(rows[:, 1:] > 1) or not np.isfinite(rows).all():
                    issue(split, sid, "label", "coordinates outside [0,1] / non-finite")
                    bad = True
                else:
                    xyxy = yolo_to_xyxy(rows, w, h)
                    tol = 1e-3 * max(w, h)
                    if np.any(xyxy[:, :2] < -tol) or np.any(xyxy[:, 2] > w + tol) or np.any(xyxy[:, 3] > h + tol):
                        issue(split, sid, "label", "box outside image")
                        bad = True
                    small = ((xyxy[:, 2] - xyxy[:, 0]) < min_side - 1e-3) | ((xyxy[:, 3] - xyxy[:, 1]) < min_side - 1e-3)
                    if small.any():
                        issue(split, sid, "label", f"{int(small.sum())} boxes with side < {min_side}px")
                        bad = True
            n_boxes += len(rows)
            per_image.append({"split": split, "sample_id": sid, "num_objects": len(rows), "valid_depth_ratio": round(ratio, 6)})
            if bad:
                invalid_data += 1
            else:
                n_valid_pairs += 1

        summary[split] = {
            "rgb": len(imgs),
            "depth": len(deps),
            "labels": len(labs),
            "valid_pairs": n_valid_pairs,
            "invalid_data": invalid_data,
            "bounding_boxes": n_boxes,
            "mean_valid_depth_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
            "expected_pairs": expected.get(split),
        }

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for sid in sorted(ids_by_split[a] & ids_by_split[b]):
            issue(f"{a}+{b}", sid, "leakage", "sample present in both splits")
    manifest = root / "meta" / "manifest.csv"
    group_leaks: list[str] = []
    if manifest.is_file():
        from src.utils.io import load_csv

        rows = load_csv(manifest)
        if rows and "group" in rows[0]:
            tg = {r["group"] for r in rows if r["split"] == "train"}
            vg = {r["group"] for r in rows if r["split"] == "val"}
            group_leaks = sorted(tg & vg)
            for g in group_leaks:
                issue("train+val", g, "group_leakage", "recording group present in train and internal val")
    else:
        issue("all", "-", "manifest", f"{manifest} missing: group leakage cannot be checked")

    def n(s):
        return summary[s]["valid_pairs"]

    checks = []
    if expected.get("train_official") is not None:
        checks.append(n("train") + n("val") == expected["train_official"])
    if expected.get("test") is not None:
        checks.append(n("test") == expected["test"])
    for s in SPLITS:  # optional per-split expectations
        if expected.get(s) is not None and s != "test":
            checks.append(n(s) == expected[s])
    if n("val") == 0:
        issue("val", "-", "split", "internal validation split is empty")
    count_ok = all(checks)
    strict = cfg["split"].get("strict_counts", True)
    passed = not issues and (count_ok or not strict)

    dist = Counter(r["num_objects"] for r in per_image)
    report = {
        "dataset_root": str(root),
        "passed": bool(passed),
        "strict_counts": strict,
        "split_counts_match_official": bool(count_ok),
        "group_leakage_train_val": len(group_leaks),
        "summary": summary,
        "total_valid_pairs": sum(s["valid_pairs"] for s in summary.values()),
        "total_bounding_boxes": sum(s["bounding_boxes"] for s in summary.values()),
        "issues_by_type": dict(Counter(i["type"] for i in issues)),
        "objects_per_image_distribution": {str(k): v for k, v in sorted(dist.items())},
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    save_csv(issues, report_dir / "integrity_issues.csv", ["split", "sample_id", "type", "detail"])
    save_csv(per_image, report_dir / "objects_per_image.csv")
    _plot_distribution(per_image, report_dir / "objects_per_image.png")
    _save_samples(root, cfg, report_dir / "samples")
    save_json(report, report_dir / "integrity_report.json")
    LOG.info("Integrity %s: %s", "PASSED" if passed else "FAILED", {k: v["valid_pairs"] for k, v in summary.items()})
    if not passed:
        LOG.error("See %s for the %d issues found", report_dir / "integrity_issues.csv", len(issues))
    return report


def _plot_distribution(per_image: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    for split in SPLITS:
        n = [r["num_objects"] for r in per_image if r["split"] == split]
        if n:
            ax.hist(n, bins=range(0, max(n) + 2), alpha=0.6, label=split)
    ax.set_xlabel("Objects per image")
    ax.set_ylabel("Images")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_samples(root: Path, cfg: dict, out_dir: Path) -> None:
    """Write RGB+boxes | depth colormap panels for a deterministic subset of samples."""
    n = int(cfg["validation"].get("num_visual_samples", 12))
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        imgs = sorted((root / "images" / split).glob("*"))
        if not imgs:
            continue
        for p in [imgs[i] for i in np.linspace(0, len(imgs) - 1, min(n // 2 or 1, len(imgs))).astype(int)]:
            im = cv2.imread(str(p))
            dep = root / "depth" / split / f"{p.stem}.png"
            lab = root / "labels" / split / f"{p.stem}.txt"
            if im is None or not dep.is_file():
                continue
            h, w = im.shape[:2]
            boxes = yolo_to_xyxy(read_yolo_label(lab), w, h)
            panel = side_by_side(
                draw_boxes(im, boxes, ["objek"] * len(boxes)),
                colorize_depth(load_metric_depth_png(dep, cfg["depth"]["png_scale"]), cfg["depth"]["max_depth"]),
            )
            cv2.imwrite(str(out_dir / f"{split}_{p.stem}.jpg"), panel)


def require_dataset_ok(report_path: str | Path = "outputs/dataset/integrity_report.json") -> dict:
    """Gate used by every training script: raise unless dataset validation has passed."""
    p = resolve_path(report_path)
    if not p.is_file():
        raise RuntimeError(f"Dataset integrity report not found ({p}). Run: python scripts/data/validate_dataset.py")
    rep = load_json(p)
    if not rep.get("passed"):
        raise RuntimeError(f"Dataset integrity check FAILED ({p}); fix the issues before training.")
    return rep


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit code 1 when validation fails."""
    ap = argparse.ArgumentParser(description="Validate the prepared SUN RGB-D dataset")
    ap.add_argument("--config", default="configs/dataset/sunrgbd.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    rep = validate(apply_overrides(load_config(args.config), args.set))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
