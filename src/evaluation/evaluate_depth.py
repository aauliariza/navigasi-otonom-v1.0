"""Evaluate every monocular depth configuration (proposal Sub-bab 3.2.5.2).

For each model's (log-affine calibrated) ``weights/best.pt``: every val image is stretch-resized to ``imgsz``,
predicted, the 1/4-resolution metric depth is bilinearly upsampled to the original ground-truth resolution, and
Eq. (3.34)-(3.38) are computed over valid pixels (0.001 <= D_gt <= 10 m). No per-image median alignment is
applied: the metrics measure the calibrated metric output, which is what the integrated system uses.

Outputs (``output_dir``): ``<name>/metrics.json``, ``<name>/per_image.csv``, ``<name>/samples/*.jpg`` (RGB | GT |
prediction | error map), ``<name>/depth_maps/*_{pred,gt}.npy``, ``depth_results.csv`` / ``.json`` (EWM-SAW decision matrix input), ``hardware.json``.

Usage::

    python -m src.evaluation.evaluate_depth --config configs/evaluation/depth.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

import models  # noqa: F401
from models.depth.yolo26_depth import load_model, predict_depth
from src.data.prepare_depth import load_metric_depth_png
from src.evaluation.benchmark import benchmark_model, hardware_description
from src.evaluation.evaluate_detection import copy_into_experiment, safe_name
from src.evaluation.metrics import DepthMetricAccumulator
from src.utils.checkpoint import best_weights, load_ultralytics_ckpt, require_file
from src.utils.config import apply_overrides, load_config, resolve_path
from src.utils.io import save_csv, save_json
from src.utils.logger import get_logger
from src.utils.visualization import colorize_depth, colorize_error, side_by_side

LOG = get_logger("evaluate_depth")


def _device(ev: dict) -> str:
    return "cpu" if str(ev["device"]) == "cpu" else f"cuda:{ev['device']}"


def evaluate_depth_model(model, image_dir: Path, depth_dir: Path, imgsz: int, min_depth: float, max_depth: float,
                         sample_dir: Path | None = None, num_samples: int = 0, save_dir: Path | None = None,
                         num_save: int = 0) -> tuple[dict, list[dict]]:
    """Return (pooled + per-image-mean metrics, per-image rows).

    ``sample_dir`` receives ``num_samples`` panels RGB | GT | prediction | abs-rel error map; ``save_dir`` receives
    ``num_save`` pairs ``<id>_pred.npy`` / ``<id>_gt.npy`` (metres, float32, GT resolution, 0 = invalid GT).
    """
    acc = DepthMetricAccumulator(min_depth, max_depth)
    images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    keep = set(np.linspace(0, len(images) - 1, num_samples).astype(int).tolist()) if num_samples and images else set()
    keep_npy = set(np.linspace(0, len(images) - 1, num_save).astype(int).tolist()) if num_save and images else set()
    for i, p in enumerate(images):
        bgr = cv2.imread(str(p))
        gt = load_metric_depth_png(depth_dir / f"{p.stem}.png")
        pred = predict_depth(model, bgr, imgsz, out_size=gt.shape)
        acc.update(pred, gt, p.stem)
        if sample_dir is not None and i in keep:
            sample_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sample_dir / f"{p.stem}.jpg"),
                        side_by_side(bgr, colorize_depth(gt, max_depth), colorize_depth(pred, max_depth), colorize_error(pred, gt)))
        if save_dir is not None and i in keep_npy:
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / f"{p.stem}_pred.npy", pred.astype(np.float32))
            np.save(save_dir / f"{p.stem}_gt.npy", gt.astype(np.float32))
    return acc.compute(), acc.per_image


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Evaluate all depth models")
    ap.add_argument("--config", default="configs/evaluation/depth.yaml")
    ap.add_argument("--include-extra", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    ev = apply_overrides(load_config(args.config), args.set)
    out_dir = resolve_path(str(ev["output_dir"]).format(split=ev["split"]))
    data = yaml.safe_load(resolve_path(ev["data_yaml"]).read_text())
    root = Path(data["path"])
    image_dir = root / data[ev["split"]]
    depth_dir = root / data[ev["split"]].replace("images", "depth", 1)
    rows = []
    entries = list(ev["models"]) + (list(ev.get("extra_models", [])) if args.include_extra else [])
    for m in entries:
        exp = apply_overrides(load_config(m["config"]), ev.get("experiment_overrides", []))
        w = require_file(best_weights(exp), f"best.pt of {m['name']}", f"Train it first: {m['config']}")
        cal = load_ultralytics_ckpt(w).get("log_affine_calibration")
        if cal is None:
            LOG.warning("%s has no log-affine calibration record; evaluating its stored (a, b)", m["name"])
        model = load_model(w, _device(ev))
        sub = out_dir / safe_name(m["name"])
        metrics, per_image = evaluate_depth_model(model, image_dir, depth_dir, ev["imgsz"], ev["min_depth"], ev["max_depth"],
                                                  sub / "samples", ev.get("num_visual_samples", 0),
                                                  sub / "depth_maps", int(ev.get("save_predictions", 0)))
        bm = benchmark_model(model, ev["imgsz"], _device(ev), **ev["benchmark"])
        row = {"model": m["name"], "experiment_id": exp["experiment"]["id"], "split": ev["split"], "weights": str(w),
               "role": "extra" if m in ev.get("extra_models", []) else "alternative",
               **{k: metrics[k] for k in ("delta1", "delta2", "delta3", "abs_rel", "sq_rel", "rmse", "rmse_log")},
               "fps": bm["fps"], "latency_ms": bm["latency_ms_mean"], "params": bm["params_total"], "gflops": bm["gflops"],
               "gpu_memory_gb": bm["gpu_memory_peak_gb"], "cal_a": (cal or {}).get("a"), "cal_b": (cal or {}).get("b")}
        save_json({"summary": row, "metrics": metrics, "calibration": cal, "benchmark": bm}, sub / "metrics.json")
        save_csv(per_image, sub / "per_image.csv")
        copy_into_experiment(sub / "metrics.json", exp, ev["split"])
        LOG.info("%s: d1=%.4f abs_rel=%.4f rmse=%.4f FPS=%.1f", m["name"], row["delta1"], row["abs_rel"], row["rmse"], row["fps"])
        rows.append(row)
    save_csv(rows, out_dir / "depth_results.csv")
    save_json(rows, out_dir / "depth_results.json")
    save_json(hardware_description(_device(ev)), out_dir / "hardware.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
