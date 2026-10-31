"""Navigation-decision accuracy (proposal Sub-bab 3.2.5.4, Eq. 3.45).

``gt_source: scenarios``: indoor test scenarios recorded by the researcher, CSV columns
``scenario_id,image,gt_direction[,description]`` with ``gt_direction`` in FORWARD / TURN_LEFT / TURN_RIGHT / STOP,
decided from the real object layout and measured distances of each scenario (docs/EXPERIMENTS.md).

``gt_source: sunrgbd``: a reference direction is derived for every val image by applying the SAME sector rule
to the ground-truth boxes with distances from the ground-truth depth (confidence 1.0, so the low-confidence rule
never fires for GT). This isolates the error introduced by the learned models; it is a derived reference, not a
human-annotated navigation label.

Outputs: ``navigation_per_scenario.csv`` (status per sector FL..FR, predicted / true direction, correct),
``navigation_summary.json`` / ``.csv`` (scenarios, correct decisions, accuracy %), ``confusion_matrix.csv``.

Usage::

    python -m src.evaluation.evaluate_navigation --config configs/evaluation/navigation.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from src.data.prepare_annotations import read_yolo_label, yolo_to_xyxy
from src.data.prepare_depth import load_metric_depth_png
from src.evaluation.evaluate_distance import gt_distance
from src.evaluation.metrics import navigation_accuracy
from src.integration.detector_depth_fusion import PerceptionPipeline
from src.navigation.direction_decision import DIRECTIONS
from src.utils.config import apply_overrides, load_config, resolve_path
from src.utils.io import load_csv, save_csv, save_json
from src.utils.logger import get_logger

LOG = get_logger("evaluate_navigation")


def sector_string(sectors: list[dict]) -> str:
    return ", ".join(f"{s['name']}={'blocked' if s['blocked'] else 'free'}" for s in sectors)


def reference_from_gt(pipe: PerceptionPipeline, gt_boxes, gt_depth, w: int, h: int) -> dict:
    """Apply the decision rule to GT boxes + GT-depth distances."""
    dc = pipe.cfg["distance"]
    objs = []
    for i, b in enumerate(gt_boxes):
        rec = gt_distance(gt_depth, b, dc)
        objs.append({"id": i, "bbox": [float(v) for v in b], "confidence": 1.0, "fusion_confidence": 1.0, **rec})
    return pipe.navigate(objs, w, h)


def run(ev: dict) -> dict:
    """Evaluate and write all artefacts; returns the summary."""
    scfg = apply_overrides(load_config(ev["system_config"]), ev.get("system_overrides", []))
    pipe = PerceptionPipeline(scfg)
    rows = []
    if ev["gt_source"] == "scenarios":
        csv_path = resolve_path(ev["scenarios_csv"])
        for sc in load_csv(csv_path):
            img = Path(sc["image"]) if Path(sc["image"]).is_absolute() else csv_path.parent / sc["image"]
            bgr = cv2.imread(str(img))
            if bgr is None:
                raise FileNotFoundError(f"Scenario image not found: {img}")
            if sc["gt_direction"] not in DIRECTIONS:
                raise ValueError(f"Scenario {sc['scenario_id']}: gt_direction must be one of {DIRECTIONS}")
            res = pipe.process(bgr)
            rows.append({"scenario_id": sc["scenario_id"], "image": sc["image"], "description": sc.get("description", ""),
                         **{f"{s['name']}": "blocked" if s["blocked"] else "free" for s in res["sectors"]}, "sector_state": sector_string(res["sectors"]),
                         "gt_direction": sc["gt_direction"], "pred_direction": res["direction"]})
    elif ev["gt_source"] == "sunrgbd":
        data = yaml.safe_load(resolve_path(ev["data_yaml"]).read_text())
        root = Path(data["path"])
        img_dir, sub = root / data[ev["split"]], data[ev["split"]]
        images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if ev.get("max_images"):
            images = images[: int(ev["max_images"])]
        for p in images:
            bgr = cv2.imread(str(p))
            h, w = bgr.shape[:2]
            gt_boxes = yolo_to_xyxy(read_yolo_label(root / sub.replace("images", "labels", 1) / f"{p.stem}.txt"), w, h)
            gt_depth = load_metric_depth_png(root / sub.replace("images", "depth", 1) / f"{p.stem}.png")
            ref = reference_from_gt(pipe, gt_boxes, gt_depth, w, h)
            res = pipe.process(bgr)
            rows.append({"scenario_id": p.stem, "image": p.name, "description": "SUN RGB-D val (GT-derived reference)",
                         **{f"{s['name']}": "blocked" if s["blocked"] else "free" for s in res["sectors"]}, "sector_state": sector_string(res["sectors"]),
                         "gt_sector_state": sector_string(ref["sectors"]), "gt_direction": ref["direction"], "pred_direction": res["direction"]})
    else:
        raise ValueError("gt_source must be 'scenarios' or 'sunrgbd'")
    for r in rows:
        r["correct"] = r["pred_direction"] == r["gt_direction"]
    acc = navigation_accuracy([r["pred_direction"] for r in rows], [r["gt_direction"] for r in rows])
    cm = [{"gt\\pred": g, **{p: sum(r["gt_direction"] == g and r["pred_direction"] == p for r in rows) for p in DIRECTIONS}} for g in DIRECTIONS]
    summary = {"gt_source": ev["gt_source"], **acc, "weights": pipe.weights}
    out = resolve_path(ev["output_dir"])
    save_csv(rows, out / "navigation_per_scenario.csv")
    save_csv(cm, out / "confusion_matrix.csv")
    save_json(summary, out / "navigation_summary.json")
    save_csv([summary], out / "navigation_summary.csv")
    LOG.info("Navigation accuracy: %d/%d = %.2f%%", acc["n_correct"], acc["n_total"], acc["accuracy_pct"])
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Evaluate navigation decision accuracy")
    ap.add_argument("--config", default="configs/evaluation/navigation.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    run(apply_overrides(load_config(args.config), args.set))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
