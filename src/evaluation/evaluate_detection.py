"""Evaluate every detection configuration (proposal Sub-bab 3.2.5.1).

For each model listed in configs/evaluation/detection.yaml (its ``weights/best.pt``):
* Ultralytics validation on the val split -> mAP50, mAP50-95 (+ P/R at the max-F1 confidence, reported as
  ``*_maxf1``);
* operating-point P / R / F1 (conf 0.25, NMS IoU 0.45, max 100 det, match IoU 0.5) with exact TP/FP/FN counts;
* model-only benchmark: FPS, latency, params, GFLOPs, peak GPU memory (src/evaluation/benchmark.py).

Outputs (``output_dir``): ``<name>/metrics.json``, ``detection_results.csv`` / ``.json`` (one row per model, the
EWM-SAW decision matrix input), ``hardware.json``.

Usage::

    python -m src.evaluation.evaluate_detection --config configs/evaluation/detection.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

import models  # noqa: F401
from models.depth.yolo26_depth import load_model
from src.data.prepare_annotations import read_yolo_label, yolo_to_xyxy
from src.evaluation.benchmark import benchmark_model, hardware_description
from src.evaluation.metrics import f1_score, match_detections, precision_recall_f1
from src.utils.checkpoint import best_weights, require_file
from src.utils.config import apply_overrides, load_config, resolve_path
from src.utils.io import save_csv, save_json
from src.utils.logger import get_logger

LOG = get_logger("evaluate_detection")


def safe_name(name: str) -> str:
    """Folder-safe version of a model display name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def copy_into_experiment(metrics_json: Path, exp: dict, split: str) -> None:
    """Keep a copy of the evaluation result inside the experiment folder (outputs/<task>/<ID>/evaluation/<split>/)."""
    import shutil

    from src.utils.checkpoint import experiment_dir

    dst = experiment_dir(exp) / "evaluation" / split
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_json, dst / "metrics.json")


def operating_point(yolo, image_dir: Path, label_dir: Path, ev: dict) -> dict:
    """Exact TP/FP/FN, P, R, F1 of the detections used at inference time."""
    op = ev["operating_point"]
    tp = fp = fn = 0
    images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for res in yolo.predict(source=[str(p) for p in images], imgsz=ev["imgsz"], conf=op["conf"], iou=op["iou_nms"], max_det=op["max_det"],
                            device=ev["device"], stream=True, verbose=False):
        h, w = res.orig_shape
        gt = yolo_to_xyxy(read_yolo_label(label_dir / f"{Path(res.path).stem}.txt"), w, h)
        pred = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros(0)
        a, b, c = match_detections(pred, scores, gt, op["match_iou"])
        tp, fp, fn = tp + a, fp + b, fn + c
    return {"tp": tp, "fp": fp, "fn": fn, **precision_recall_f1(tp, fp, fn), "images": len(images)}


def evaluate_one(name: str, weights: Path, ev: dict, out_dir: Path) -> dict:
    """Full evaluation of one checkpoint."""
    import yaml
    from ultralytics import YOLO

    data = yaml.safe_load(resolve_path(ev["data_yaml"]).read_text())
    root = Path(data["path"])
    yolo = YOLO(str(weights))
    end2end = bool(ev["operating_point"].get("end2end", False))
    yolo.model.end2end = end2end  # explicit Detect-head mode (keputusan D-6), identical for mAP and operating point
    val = yolo.val(data=str(resolve_path(ev["data_yaml"])), split=ev["split"], imgsz=ev["imgsz"], batch=ev["batch"], conf=ev["val_conf"],
                   device=ev["device"], plots=True, project=str(out_dir), name=safe_name(name), exist_ok=True, verbose=False)
    box = val.box
    row = {"model": name, "weights": str(weights), "detection_head": "one2one_nms_free" if end2end else "one2many_nms", "mAP50": float(box.map50), "mAP50-95": float(box.map),
           "precision_maxf1": float(box.mp), "recall_maxf1": float(box.mr), "f1_maxf1": f1_score(float(box.mp), float(box.mr))}
    op = operating_point(yolo, root / data[ev["split"]], root / data[ev["split"]].replace("images", "labels", 1), ev)
    row.update(precision=op["precision"], recall=op["recall"], f1=op["f1"], tp=op["tp"], fp=op["fp"], fn=op["fn"])
    bm = benchmark_model(load_model(weights, "cpu" if str(ev["device"]) == "cpu" else f"cuda:{ev['device']}"), ev["imgsz"],
                         "cpu" if str(ev["device"]) == "cpu" else f"cuda:{ev['device']}", **ev["benchmark"])
    row.update(fps=bm["fps"], latency_ms=bm["latency_ms_mean"], params=bm["params_total"], gflops=bm["gflops"],
               gpu_memory_gb=bm["gpu_memory_peak_gb"])
    save_json({"summary": row, "operating_point": op, "ultralytics_speed_ms": val.speed, "benchmark": bm}, out_dir / safe_name(name) / "metrics.json")
    LOG.info("%s: mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f F1=%.4f FPS=%.1f", name, row["mAP50"], row["mAP50-95"], row["precision"], row["recall"], row["f1"], row["fps"])
    return row


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Evaluate all detection models")
    ap.add_argument("--config", default="configs/evaluation/detection.yaml")
    ap.add_argument("--include-extra", action="store_true", help="Also evaluate extra_models (e.g. the CA teacher)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    ev = apply_overrides(load_config(args.config), args.set)
    out_dir = resolve_path(str(ev["output_dir"]).format(split=ev["split"]))
    entries = list(ev["models"]) + (list(ev.get("extra_models", [])) if args.include_extra else [])
    rows = []
    for m in entries:
        exp = apply_overrides(load_config(m["config"]), ev.get("experiment_overrides", []))
        w = require_file(best_weights(exp), f"best.pt of {m['name']}", f"Train it first: {m['config']}")
        row = {**evaluate_one(m["name"], w, ev, out_dir), "experiment_id": exp["experiment"]["id"], "split": ev["split"],
               "role": "extra" if m in ev.get("extra_models", []) else "alternative"}
        copy_into_experiment(out_dir / safe_name(m["name"]) / "metrics.json", exp, ev["split"])
        rows.append(row)
    save_csv(rows, out_dir / "detection_results.csv")
    save_json(rows, out_dir / "detection_results.json")
    save_json(hardware_description(ev["device"]), out_dir / "hardware.json")
    LOG.info("Saved %s", out_dir / "detection_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
