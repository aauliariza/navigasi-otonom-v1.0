"""Object-distance accuracy of the integrated system (proposal Sub-bab 3.2.5.3, Eq. 3.39-3.43).

``gt_source: sunrgbd`` (same-box protocol): for each image of the configured split (default: official test) the detector boxes are reduced with the SAME
inner-ROI / valid-filter / percentile-clip / median procedure on (a) the predicted metric depth and (b) the
SUN RGB-D ground-truth depth, so the difference only comes from the depth source. Only objects valid in both
(R_valid >= 0.30) are scored; invalid ones are counted separately.

``gt_source: measured``: distances measured physically in the indoor test environment, supplied as CSV with
columns ``image,x1,y1,x2,y2,gt_distance_m`` (box drawn around the measured object in that image). Each annotated
box is matched to a detection (IoU >= match_iou); unmatched objects are reported as ``not_detected``.

Outputs: ``distance_per_object.csv`` (true distance, estimate, |error|, error %, validity), ``distance_summary.json``
/ ``.csv`` (MAE, RMSE, MAPE, valid / invalid counts), ``distance_evaluation.png`` (estimate vs. true, error histogram).

Usage::

    python -m src.evaluation.evaluate_distance --config configs/evaluation/distance.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.data.prepare_depth import load_metric_depth_png
from src.evaluation.metrics import box_iou, distance_metrics
from src.integration.detector_depth_fusion import PerceptionPipeline
from src.integration.object_distance import estimate_distance
from src.utils.config import apply_overrides, load_config, resolve_path
from src.utils.io import load_csv, save_csv, save_json
from src.utils.logger import get_logger

LOG = get_logger("evaluate_distance")


def gt_distance(gt_depth: np.ndarray, box_img, dc: dict) -> dict:
    """Same reduction as the prediction, on GT depth at image resolution (alignment = identity)."""
    return estimate_distance(gt_depth, box_img, dc["roi_ratio"], dc["roi_min_size"], dc["min_depth"], dc["max_depth"], dc["min_valid_ratio"],
                             tuple(dc["percentiles"]), dc["clip_mode"], "axial")


def row_for(image: str, obj: dict, d_gt: float | None, gt_valid: bool, status: str) -> dict:
    d_pred = obj["distance_m"] if obj else None
    both = bool(obj and obj["valid"] and gt_valid and d_gt)
    return {"image": image, "object_id": obj["id"] if obj else "", "x1": obj["bbox"][0] if obj else "", "y1": obj["bbox"][1] if obj else "",
            "x2": obj["bbox"][2] if obj else "", "y2": obj["bbox"][3] if obj else "", "confidence": obj["confidence"] if obj else "",
            "distance_true_m": d_gt, "distance_pred_m": d_pred, "pred_valid": bool(obj and obj["valid"]), "gt_valid": gt_valid,
            "pred_valid_ratio": obj["valid_ratio"] if obj else "", "abs_error_m": abs(d_pred - d_gt) if both else "",
            "error_pct": 100 * abs(d_pred - d_gt) / d_gt if both else "", "evaluated": both, "status": status}


def run(ev: dict) -> dict:
    """Evaluate and write all artefacts; returns the summary."""
    scfg = load_config(ev["system_config"])
    scfg = apply_overrides(scfg, ev.get("system_overrides", []))
    pipe = PerceptionPipeline(scfg)
    dc = scfg["distance"]
    rows: list[dict] = []
    if ev["gt_source"] == "sunrgbd":
        data = yaml.safe_load(resolve_path(ev["data_yaml"]).read_text())
        root = Path(data["path"])
        img_dir = root / data[ev["split"]]
        dep_dir = root / data[ev["split"]].replace("images", "depth", 1)
        images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if ev.get("max_images"):
            images = images[: int(ev["max_images"])]
        for p in images:
            bgr = cv2.imread(str(p))
            gt = load_metric_depth_png(dep_dir / f"{p.stem}.png")
            res = pipe.process(bgr)
            for o in res["objects"]:
                g = gt_distance(gt, o["bbox"], dc)
                rows.append(row_for(p.name, o, g["distance_m"], g["valid"], "ok" if g["valid"] else "gt_invalid"))
    elif ev["gt_source"] == "measured":
        meas = load_csv(resolve_path(ev["measurements_csv"]))
        by_img: dict[str, list[dict]] = {}
        for m in meas:
            by_img.setdefault(m["image"], []).append(m)
        base = resolve_path(ev["measurements_csv"]).parent
        for img, items in by_img.items():
            path = Path(img) if Path(img).is_absolute() else base / img
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(f"Measurement image not found: {path}")
            res = pipe.process(bgr)
            det = np.array([o["bbox"] for o in res["objects"]]) if res["objects"] else np.zeros((0, 4))
            ann = np.array([[float(m[k]) for k in ("x1", "y1", "x2", "y2")] for m in items])
            iou = box_iou(ann, det)
            used = set()
            for k, m in enumerate(items):
                j = int(np.argmax(iou[k])) if det.size else -1
                if j >= 0 and iou[k, j] >= ev["match_iou"] and j not in used:
                    used.add(j)
                    rows.append(row_for(img, res["objects"][j], float(m["gt_distance_m"]), True, "ok"))
                else:
                    rows.append(row_for(img, {}, float(m["gt_distance_m"]), True, "not_detected"))
    else:
        raise ValueError("gt_source must be 'sunrgbd' or 'measured'")
    ok = [r for r in rows if r["evaluated"]]
    summary = {"gt_source": ev["gt_source"], "objects_total": len(rows), "objects_valid": len(ok),
               "objects_invalid_pred": sum(not r["pred_valid"] for r in rows if r["status"] != "not_detected"),
               "objects_invalid_gt": sum(not r["gt_valid"] for r in rows), "objects_not_detected": sum(r["status"] == "not_detected" for r in rows),
               **distance_metrics([r["distance_pred_m"] for r in ok], [r["distance_true_m"] for r in ok]), "weights": pipe.weights}
    out = resolve_path(ev["output_dir"])
    save_csv(rows, out / "distance_per_object.csv")
    save_json(summary, out / "distance_summary.json")
    save_csv([summary], out / "distance_summary.csv")
    if ok:
        from src.utils.visualization import plot_distance_evaluation

        plot_distance_evaluation([r["distance_pred_m"] for r in ok], [r["distance_true_m"] for r in ok], out / "distance_evaluation.png")
    LOG.info("Distance: valid=%d/%d MAE=%.4f RMSE=%.4f MAPE=%.2f%%", summary["objects_valid"], summary["objects_total"], summary["mae"], summary["rmse"], summary["mape"])
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Evaluate object distance accuracy")
    ap.add_argument("--config", default="configs/evaluation/distance.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    run(apply_overrides(load_config(args.config), args.set))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
