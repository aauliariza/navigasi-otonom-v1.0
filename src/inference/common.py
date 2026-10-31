"""Shared helpers of the three inference front-ends (image / video / webcam), prompt section 14.

* :func:`build_arg_parser` -- common CLI options (config, model checkpoints, device, output folder, overrides);
* :func:`load_pipeline`   -- builds :class:`PerceptionPipeline` from those options;
* :func:`result_rows`     -- per-object and per-frame rows for CSV export;
* :class:`FpsMeter`       -- moving-window FPS of the processing loop (displayed on video / webcam frames).

All three front-ends feed ONE monocular RGB frame to both models. The output is only a recommended direction on
the interface; no motor / wheelchair control is implemented anywhere in this repository.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

from src.integration.detector_depth_fusion import PerceptionPipeline, decision_row, objects_rows
from src.utils.config import apply_overrides, load_config, resolve_path


def build_arg_parser(description: str, default_output: str) -> argparse.ArgumentParser:
    """Arguments shared by predict_image / predict_video / webcam."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/system/inference.yaml", help="System configuration YAML")
    ap.add_argument("--detection-weights", default=None, help="YOLO26n checkpoint (default: EWM-SAW selection)")
    ap.add_argument("--depth-weights", default=None, help="YOLO26n-Depth checkpoint (default: EWM-SAW selection)")
    ap.add_argument("--device", default=None, help="'cpu' or GPU index, e.g. 0 (overrides config)")
    ap.add_argument("--output", default=default_output, help="Output folder")
    ap.add_argument("--set", nargs="*", default=[], help="Config overrides key.sub=value")
    return ap


def load_cfg(args: argparse.Namespace) -> dict:
    """System config with CLI overrides applied."""
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.detection_weights:
        cfg["models"]["detection_weights"] = args.detection_weights
    if args.depth_weights:
        cfg["models"]["depth_weights"] = args.depth_weights
    if args.device is not None:
        cfg["device"] = args.device
    return cfg


def load_pipeline(cfg: dict) -> PerceptionPipeline:
    """Build the pipeline (raises :class:`InferenceError` with module/cause/file/solution on failure)."""
    return PerceptionPipeline(cfg)


def output_dir(path: str | Path) -> Path:
    """Create and return the output folder."""
    out = resolve_path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def result_rows(name: str, result: dict, extra: dict | None = None) -> tuple[list[dict], dict]:
    """(object rows, decision row) of one processed frame, both tagged with ``extra`` columns."""
    extra = extra or {}
    objs = [{**extra, **r} for r in objects_rows(name, result)]
    return objs, {**extra, **decision_row(name, result)}


class FpsMeter:
    """FPS over the last ``window`` frames of the full processing loop (read + infer + render)."""

    def __init__(self, window: int = 30):
        self.times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Register a frame and return the current FPS (0 until two frames were seen)."""
        self.times.append(time.perf_counter())
        if len(self.times) < 2:
            return 0.0
        dt = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / dt if dt > 0 else 0.0
