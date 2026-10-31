"""Computational-efficiency measurement (proposal Sub-bab 3.2.5.1, Eq. 3.25-3.26; prompt section 13).

Three separately reported measurements:

1. ``model_only``   -- forward pass of one model on a pre-built input tensor (B=1, imgsz x imgsz). Excludes image
   decoding, resizing, normalisation, NMS/decoding to Python objects and post-processing.
2. ``pipeline``     -- forward passes of BOTH models of the integrated system on the same pre-built tensor
   (detection + depth), excluding pre/post-processing (see src/integration/detector_depth_fusion.py).
3. ``end_to_end``   -- full processing of a BGR frame: preprocessing of both models, inference, detection
   post-processing, depth upsampling, spatial alignment, inner ROI distance estimation and navigation decision.

All use warm-up iterations, ``torch.cuda.synchronize()`` around timed regions on GPU and report mean/std/median
latency in ms and FPS = frames / total time (Eq. 3.26). Peak GPU memory = ``torch.cuda.max_memory_allocated``
during the timed loop (reset before warm-up). Parameters are counted on the fused inference model; FLOPs =
2 x MACs from ``ultralytics.utils.torch_utils.get_flops`` (thop) at the evaluation input size (Eq. 3.25).
No number from the Ultralytics documentation is used as a result.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_callable(fn: Callable[[], object], device: torch.device, warmup: int = 50, iters: int = 300) -> dict:
    """Time ``fn`` (already bound to its inputs). Returns latency stats, FPS and peak GPU memory."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        _sync(device)
        times = []
        t_total = time.perf_counter()
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)
        total = time.perf_counter() - t_total
    t = np.asarray(times)
    out = {"latency_ms_mean": float(t.mean()), "latency_ms_std": float(t.std()), "latency_ms_median": float(np.median(t)),
           "latency_ms_p95": float(np.percentile(t, 95)), "fps": float(iters / total), "iters": iters, "warmup": warmup}
    out["gpu_memory_peak_gb"] = float(torch.cuda.max_memory_allocated(device) / 1e9) if device.type == "cuda" else None
    return out


def model_complexity(model: torch.nn.Module, imgsz: int) -> dict:
    """Parameters (total / trainable) and GFLOPs of an inference model."""
    from models.common.model_utils import model_stats

    return model_stats(model, imgsz)


def benchmark_model(model: torch.nn.Module, imgsz: int = 640, device="cpu", warmup: int = 50, iters: int = 300, half: bool = False) -> dict:
    """Model-only benchmark of an eval-mode (fused) model."""
    device = torch.device(device)
    model = model.to(device).eval()
    dtype = torch.float16 if half and device.type == "cuda" else torch.float32
    model = model.to(dtype)
    x = torch.rand(1, 3, imgsz, imgsz, device=device, dtype=dtype)
    res = time_callable(lambda: model(x), device, warmup, iters)
    res.update(model_complexity(model.float(), imgsz))
    res.update(mode="model_only", imgsz=imgsz, batch=1, precision="fp16" if dtype == torch.float16 else "fp32",
               includes_preprocessing=False, includes_postprocessing=False)
    return res


def hardware_description(device) -> dict:
    """Device/software description stored next to every benchmark result."""
    from src.training.trainer_utils import environment_info

    info = environment_info()
    info["benchmark_device"] = str(device)
    if str(device) == "cpu" or not torch.cuda.is_available():
        import platform

        info["cpu"] = platform.processor() or platform.machine()
        info["torch_threads"] = torch.get_num_threads()
    return info


def benchmark_system(pipe, bgr, warmup: int = 20, iters: int = 100) -> dict:
    """Integrated measurements: (2) both models forward-only, (3) end-to-end frame processing."""
    import models  # noqa: F401
    from models.depth.yolo26_depth import load_model

    device = torch.device(pipe.device)
    det = load_model(pipe.weights["detection"], device)
    dep = pipe.depth_model
    x_det = torch.rand(1, 3, pipe.cfg["detection"]["imgsz"], pipe.cfg["detection"]["imgsz"], device=device)
    x_dep = torch.rand(1, 3, pipe.cfg["depth"]["imgsz"], pipe.cfg["depth"]["imgsz"], device=device)
    pipeline = time_callable(lambda: (det(x_det), dep(x_dep)), device, warmup, iters)
    pipeline.update(mode="pipeline_models_only", includes_preprocessing=False, includes_postprocessing=False)
    e2e = time_callable(lambda: pipe.process(bgr), device, warmup, iters)
    e2e.update(mode="end_to_end", includes_preprocessing=True, includes_postprocessing=True,
               stages="detector preprocess+inference+postprocess, depth preprocess+inference, alignment, inner ROI distance, sectors, decision")
    return {"pipeline_models_only": pipeline, "end_to_end": e2e}


def main(argv: list[str] | None = None) -> int:
    """CLI: benchmark the integrated system on one image -> outputs/integrated_system/benchmark_system.json."""
    import argparse

    import cv2

    from src.integration.detector_depth_fusion import PerceptionPipeline
    from src.utils.config import apply_overrides, load_config, resolve_path
    from src.utils.io import save_json

    ap = argparse.ArgumentParser(description="Benchmark the integrated system (pipeline + end-to-end latency)")
    ap.add_argument("--config", default="configs/system/inference.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    pipe = PerceptionPipeline(cfg)
    bgr = cv2.imread(str(resolve_path(args.image)))
    if bgr is None:
        raise FileNotFoundError(args.image)
    res = {"weights": pipe.weights, "image_shape": list(bgr.shape), **benchmark_system(pipe, bgr, args.warmup, args.iters),
           "hardware": hardware_description(pipe.device)}
    path = save_json(res, resolve_path(cfg["output_dir"]) / "benchmark_system.json")
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
