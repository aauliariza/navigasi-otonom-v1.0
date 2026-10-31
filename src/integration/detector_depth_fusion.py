"""Integrated perception + navigation decision from ONE monocular RGB frame (proposal Sub-bab 3.2.4.5-3.2.4.7).

    RGB frame ──► YOLO26n (KD)        ──► boxes, confidence (conf 0.25, NMS IoU 0.45, max 100)
              └─► YOLO26n-Depth (KD)  ──► metric depth map (log-affine calibrated)
    boxes ──► spatial alignment (Eq. 3.11-3.12) ──► inner ROI 60% ──► valid filter, R_valid, p5-p95 clip, median
          ──► per-object distance (axial) ──► global ROI 60% ──► 5 sectors ──► blocked/free ──► direction

No depth sensor, LiDAR or stereo input is used at inference; the output is an interface-level decision only.

Usage (images or a folder)::

    python -m src.integration.detector_depth_fusion --config configs/system/inference.yaml --source path/to/images

Outputs (``output_dir``): ``<image>_result.jpg`` (boxes + distance labels + sectors + direction | depth map),
``objects.csv``, ``decisions.csv`` and ``results.json``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import models  # noqa: F401
from models.depth.yolo26_depth import load_model, preprocess
from src.integration.object_distance import estimate_distance
from src.integration.spatial_alignment import scale_boxes_to_depth, scale_intrinsics
from src.navigation.direction_decision import decide_direction
from src.navigation.navigation_roi import global_navigation_roi
from src.navigation.obstacle_analysis import BlockingRules, analyse_sectors
from src.navigation.sector_division import divide_sectors
from src.utils.config import apply_overrides, load_config, resolve_path
from src.utils.io import load_json, save_csv, save_json
from src.utils.logger import get_logger
from src.utils.visualization import colorize_depth, draw_boxes, draw_sectors, draw_status, side_by_side

LOG = get_logger("integration")


class InferenceError(RuntimeError):
    """Informative failure of the inference system: module, cause, related file and suggested fix."""

    def __init__(self, module: str, cause: str, file: str | Path | None = None, solution: str | None = None):
        self.module, self.cause, self.file, self.solution = module, cause, file, solution
        msg = f"[{module}] {cause}"
        if file is not None:
            msg += f"\n  File   : {file}"
        if solution:
            msg += f"\n  Solusi : {solution}"
        super().__init__(msg)


def resolve_model_paths(cfg: dict) -> tuple[Path, Path]:
    """Detection/depth checkpoints: explicit paths, else the EWM-SAW selection file. Both must exist."""
    m = cfg["models"]
    det, dep = m.get("detection_weights"), m.get("depth_weights")
    if not (det and dep):
        sel_path = resolve_path(m["selected_models"])
        if not sel_path.is_file():
            raise InferenceError("model_loader", "file hasil pemilihan model (EWM-SAW) tidak ditemukan", sel_path,
                                 "jalankan `python -m src.optimization.model_selection` atau isi models.detection_weights dan "
                                 "models.depth_weights (CLI: --detection-weights / --depth-weights)")
        sel = load_json(sel_path)
        det, dep = det or sel["detection"]["weights"], dep or sel["depth"]["weights"]
    det, dep = resolve_path(det), resolve_path(dep)
    for what, p in (("deteksi", det), ("depth", dep)):
        if not p.is_file():
            raise InferenceError("model_loader", f"checkpoint model {what} tidak ditemukan", p,
                                 "latih eksperimen terkait terlebih dahulu (README bagian training) atau perbaiki path checkpoint")
    return det, dep


def _device(cfg_device) -> str:
    """Torch device string; falls back to CPU (with a warning) when CUDA is requested but unavailable."""
    d = str(cfg_device)
    if d == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        LOG.warning("[device] GPU '%s' diminta tetapi CUDA tidak tersedia; inferensi dijalankan di CPU (FPS akan lebih rendah).", d)
        return "cpu"
    return d if d.startswith("cuda") else f"cuda:{d}"


class PerceptionPipeline:
    """Holds both models and runs the full decision chain on BGR frames."""

    def __init__(self, cfg: dict, detection_weights: str | Path | None = None, depth_weights: str | Path | None = None):
        from ultralytics import YOLO

        self.cfg = cfg
        if detection_weights is None or depth_weights is None:
            detection_weights, depth_weights = resolve_model_paths(cfg)
        self.device = _device(cfg.get("device", "cpu"))
        try:
            self.detector = YOLO(str(detection_weights))
        except Exception as e:  # re-raised with context, never swallowed
            raise InferenceError("detection_model", f"gagal memuat model deteksi: {type(e).__name__}: {e}", detection_weights,
                                 "pastikan file adalah checkpoint Ultralytics YOLO26 (weights/best.pt) yang utuh") from e
        try:
            self.depth_model = load_model(depth_weights, self.device)
        except Exception as e:
            raise InferenceError("depth_model", f"gagal memuat model depth: {type(e).__name__}: {e}", depth_weights,
                                 "pastikan file adalah checkpoint YOLO26-Depth (weights/best.pt) yang utuh") from e
        if getattr(self.detector, "task", "detect") != "detect":
            raise InferenceError("detection_model", f"checkpoint deteksi ber-task '{self.detector.task}', bukan 'detect'", detection_weights,
                                 "tukar --detection-weights dan --depth-weights bila tertukar")
        # Detect-head mode (keputusan D-6): False = one-to-many head + NMS (IoU 0.45, proposal Sub-bab 3.2.4.1; the
        # Ultralytics 8.4.194 default for .pt checkpoints); True = NMS-free one-to-one head (IoU threshold unused).
        self.end2end = bool(cfg["detection"].get("end2end", False))
        self.detector.model.end2end = self.end2end
        head = self.depth_model.model[-1]
        if not hasattr(head, "cal_a"):
            raise InferenceError("depth_model", "checkpoint depth tidak memiliki Depth head (cal_a/cal_b)", depth_weights,
                                 "gunakan checkpoint hasil train_depth / train_depth_kd")
        n = cfg["navigation"]
        self.rules = BlockingRules(n["min_overlap"], n["block_distance"], n["invalid_depth_blocks"], n["low_confidence_blocks"], n["low_conf_threshold"])
        self.weights = {"detection": str(detection_weights), "depth": str(depth_weights),
                        "detection_head": "one2one_nms_free" if cfg["detection"].get("end2end", False) else "one2many_nms"}

    # --------------------------------------------------------------------------------------------------------- stages
    def detect(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Boxes (N, 4) xyxy in image pixels and confidences (N,)."""
        d = self.cfg["detection"]
        r = self.detector.predict(bgr, imgsz=d["imgsz"], conf=d["conf"], iou=d["iou"], max_det=d["max_det"], device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        return r.boxes.xyxy.cpu().numpy().astype(np.float64), r.boxes.conf.cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def estimate_depth(self, bgr: np.ndarray) -> np.ndarray:
        """Metric depth map (model resolution, or image resolution when ``depth.resolution: image``)."""
        d = self.depth_model(preprocess(bgr, self.cfg["depth"]["imgsz"], self.device))
        d = d[0] if isinstance(d, (tuple, list)) else d
        if self.cfg["depth"].get("resolution", "model") == "image":
            d = torch.nn.functional.interpolate(d.float(), size=bgr.shape[:2], mode="bilinear", align_corners=True)
        return d[0, 0].float().cpu().numpy()

    def distances(self, boxes: np.ndarray, conf: np.ndarray, depth: np.ndarray, image_hw: tuple[int, int]) -> list[dict]:
        """Per-object distance records (Sub-bab 3.2.4.6)."""
        dc = self.cfg["distance"]
        boxes_d = scale_boxes_to_depth(boxes, image_hw, depth.shape[:2])
        K = None
        if dc.get("mode", "axial") == "euclidean":
            if dc.get("intrinsics") is None:
                raise ValueError("distance.mode=euclidean requires distance.intrinsics")
            K = scale_intrinsics(np.asarray(dc["intrinsics"]), image_hw, depth.shape[:2])
        objs = []
        for i, (b, bd, c) in enumerate(zip(boxes, boxes_d, conf)):
            rec = estimate_distance(depth, bd, dc["roi_ratio"], dc["roi_min_size"], dc["min_depth"], dc["max_depth"], dc["min_valid_ratio"],
                                    tuple(dc["percentiles"]), dc["clip_mode"], dc.get("mode", "axial"), K)
            fusion = float(c) * (rec["valid_ratio"] if self.cfg["fusion"]["confidence"] == "detection_x_valid_ratio" else 1.0)
            objs.append({"id": i, "bbox": [float(v) for v in b], "bbox_depth": [float(v) for v in bd], "confidence": float(c),
                         "fusion_confidence": fusion, **rec})
        return objs

    def navigate(self, objects: list[dict], width: int, height: int) -> dict:
        """Global ROI, sectors, blocked/free and direction (Sub-bab 3.2.4.7)."""
        n = self.cfg["navigation"]
        roi = global_navigation_roi(width, height, n["roi_width_ratio"])
        sectors = analyse_sectors(objects, divide_sectors(roi, n["sectors"]), roi, self.rules)
        command, chosen = decide_direction(sectors, n["priority"])
        return {"roi": list(roi), "sectors": sectors, "direction": command, "chosen_sector": chosen}

    # --------------------------------------------------------------------------------------------------------- full frame
    def process(self, bgr: np.ndarray) -> dict:
        """Run every stage on one BGR frame; timings are wall-clock ms (GPU synchronised)."""
        if bgr is None or not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.size == 0:
            raise InferenceError("pipeline", f"frame tidak valid (shape={getattr(bgr, 'shape', None)}); dibutuhkan citra BGR HxWx3",
                                 solution="periksa sumber gambar/video/kamera")
        sync = (lambda: torch.cuda.synchronize()) if self.device.startswith("cuda") else (lambda: None)
        t0 = time.perf_counter()
        boxes, conf = self.detect(bgr)
        sync()
        t1 = time.perf_counter()
        depth = self.estimate_depth(bgr)
        sync()
        t2 = time.perf_counter()
        objects = self.distances(boxes, conf, depth, bgr.shape[:2])
        nav = self.navigate(objects, bgr.shape[1], bgr.shape[0])
        t3 = time.perf_counter()
        return {"objects": objects, "depth": depth, **nav,
                "timing_ms": {"detection": (t1 - t0) * 1e3, "depth": (t2 - t1) * 1e3, "fusion_navigation": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3}}

    @staticmethod
    def render(bgr: np.ndarray, result: dict, max_depth: float = 10.0, status_lines: list[str] | None = None) -> np.ndarray:
        """Visualisation: RGB with boxes, distance labels, ROI/sectors, direction and status text | depth colormap."""
        labels, colors = [], []
        for o in result["objects"]:
            dist = f"{o['distance_m']:.2f} m" if o["valid"] else "invalid"
            labels.append(f"{o['confidence']:.2f} | {dist}")
            colors.append((0, 200, 255) if o["valid"] else (160, 160, 160))
        vis = draw_boxes(bgr, [o["bbox"] for o in result["objects"]], labels, colors)
        vis = draw_sectors(vis, result["sectors"], result["direction"])
        lines = list(status_lines or []) + [f"latency {result['timing_ms']['total']:.0f} ms"]
        vis = draw_status(vis, lines)
        return side_by_side(vis, colorize_depth(result["depth"], max_depth), height=bgr.shape[0])


def iter_sources(source: str | Path) -> list[Path]:
    """Image files from a file or folder."""
    p = resolve_path(source)
    if p.is_dir():
        return sorted(x for x in p.glob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    return [p]


def objects_rows(name: str, result: dict) -> list[dict]:
    """Flat per-object rows for CSV export."""
    return [{"image": name, "object_id": o["id"], "x1": o["bbox"][0], "y1": o["bbox"][1], "x2": o["bbox"][2], "y2": o["bbox"][3],
             "confidence": o["confidence"], "fusion_confidence": o["fusion_confidence"], "distance_m": o["distance_m"],
             "valid": o["valid"], "valid_ratio": o["valid_ratio"], "n_valid": o["n_valid"], "n_total": o["n_total"], "reason": o["reason"]}
            for o in result["objects"]]


def decision_row(name: str, result: dict) -> dict:
    """Flat per-frame decision row."""
    row = {"image": name, "direction": result["direction"], "chosen_sector": result["chosen_sector"] or ""}
    for s in result["sectors"]:
        row[f"{s['name']}_status"] = "blocked" if s["blocked"] else "free"
        row[f"{s['name']}_blockers"] = ";".join(f"{b['index']}:{b['reason']}" for b in s["blockers"])
    row.update({f"time_{k}_ms": v for k, v in result["timing_ms"].items()})
    return row


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Run the integrated detection + depth + navigation system on images")
    ap.add_argument("--config", default="configs/system/inference.yaml")
    ap.add_argument("--source", required=True, help="Image file or folder")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    pipe = PerceptionPipeline(cfg)
    out = resolve_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    objs, decs, full = [], [], []
    for p in iter_sources(args.source):
        bgr = cv2.imread(str(p))
        if bgr is None:
            LOG.warning("Unreadable image %s", p)
            continue
        res = pipe.process(bgr)
        cv2.imwrite(str(out / f"{p.stem}_result.jpg"), pipe.render(bgr, res, cfg["distance"]["max_depth"]))
        objs += objects_rows(p.name, res)
        decs.append(decision_row(p.name, res))
        full.append({"image": p.name, **{k: v for k, v in res.items() if k != "depth"}})
        LOG.info("%s: %d objects -> %s", p.name, len(res["objects"]), res["direction"])
    save_csv(objs, out / "objects.csv")
    save_csv(decs, out / "decisions.csv")
    save_json({"weights": pipe.weights, "config": cfg, "results": full}, out / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
