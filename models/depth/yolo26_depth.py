"""YOLO26-Depth helpers: inference on an RGB image and post-training log-affine calibration.

Architecture: official Ultralytics ``yolo26{n,x}-depth.yaml`` (backbone/neck of YOLO26 + ``Depth`` head on
P3/P4/P5 producing log-depth at 1/4 resolution, ``depth = exp(a * log d + b)`` with calibration buffers
``cal_a``/``cal_b``). Optional CBAM teacher via models/common/model_utils.attention_yaml.

Log-affine calibration (Eq. 2.4, proposal Sub-bab 3.2.4.3): ``D_metric = exp(a * log D_pred + b)``. ``(a, b)`` are
fitted in closed form by least squares on ``log D_gt = a * log D_pred + b`` over valid pixels (0.001-10 m) of a
calibration set, then written into the checkpoint's ``cal_a``/``cal_b`` buffers so every later use of the
checkpoint (evaluation, integration, interface) outputs metric depth. The calibration set is the INTERNAL
VALIDATION split (``calibration.split: val``, carved out of the official alltrain, src/data/create_split.py), never
the official test split, as required by the prompt (section 10). Compared with the Ultralytics automatic
calibration (scale-only, a fixed to 1), the full two-parameter log-affine form of Eq. (2.4) is fitted.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.data.prepare_depth import load_metric_depth_png
from src.utils.checkpoint import load_ultralytics_ckpt, save_ultralytics_ckpt
from src.utils.io import save_json


def preprocess(bgr: np.ndarray, imgsz: int, device, half: bool = False) -> torch.Tensor:
    """Stretch-resize to ``imgsz`` x ``imgsz`` (same as Ultralytics depth val/predict ``scale_fill``), RGB, [0, 1]."""
    im = cv2.resize(bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    t = torch.from_numpy(np.ascontiguousarray(im)).to(device).permute(2, 0, 1).float().div_(255.0)[None]
    return t.half() if half else t


def load_model(path: str | Path, device="cpu", fuse: bool = True):
    """Load an Ultralytics checkpoint as an eval-mode ``nn.Module`` (Detection or Depth model)."""
    import models  # noqa: F401  (attention modules must be importable for unpickling)

    model = load_ultralytics_ckpt(path)["model"].float().to(device).eval()
    if fuse and hasattr(model, "fuse"):
        model = model.fuse(verbose=False)
    return model


def depth_head(model):
    """Return the Depth head (holder of cal_a/cal_b)."""
    return model.model[-1]


@torch.no_grad()
def predict_depth(model, bgr: np.ndarray, imgsz: int = 640, out_size: tuple[int, int] | None = None) -> np.ndarray:
    """Metric depth map (metres, float32). ``out_size=(H, W)`` bilinearly resizes the 1/4-resolution output."""
    device = next(model.parameters()).device
    d = model(preprocess(bgr, imgsz, device))
    d = d[0] if isinstance(d, (tuple, list)) else d
    if out_size is not None and tuple(d.shape[-2:]) != tuple(out_size):
        d = F.interpolate(d.float(), size=out_size, mode="bilinear", align_corners=True)
    return d[0, 0].float().cpu().numpy()


def fit_log_affine(log_pred: np.ndarray, log_gt: np.ndarray) -> tuple[float, float, str]:
    """Closed-form least squares for ``log_gt = a * log_pred + b``.

    A non-positive slope would invert the depth ordering (a degenerate, e.g. untrained, model); in that case the
    scale-only solution ``a = 1, b = mean(log_gt - log_pred)`` is returned instead and flagged.

    Returns:
        (a, b, mode) with mode ``log_affine`` or ``scale_only_fallback``.
    """
    A = np.stack([log_pred, np.ones_like(log_pred)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, log_gt, rcond=None)
    if not np.isfinite(a) or a <= 0:
        return 1.0, float(np.mean(log_gt - log_pred)), "scale_only_fallback"
    return float(a), float(b), "log_affine"


def collect_calibration_pixels(model, images: list[Path], depth_dir: Path, imgsz: int, min_depth: float, max_depth: float,
                               pixels_per_image: int = 20000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Run the uncalibrated model and gather (log_pred, log_gt) of valid pixels at GT resolution."""
    rng = np.random.default_rng(seed)
    head = depth_head(model)
    a0, b0 = head.cal_a.clone(), head.cal_b.clone()
    head.cal_a.fill_(1.0)
    head.cal_b.fill_(0.0)
    lp, lg = [], []
    try:
        for img in images:
            bgr = cv2.imread(str(img))
            gt = load_metric_depth_png(depth_dir / f"{img.stem}.png")
            pred = predict_depth(model, bgr, imgsz, out_size=gt.shape)
            m = np.isfinite(gt) & (gt >= min_depth) & (gt <= max_depth) & np.isfinite(pred) & (pred > 0)
            idx = np.flatnonzero(m)
            if idx.size == 0:
                continue
            if idx.size > pixels_per_image:
                idx = rng.choice(idx, pixels_per_image, replace=False)
            lp.append(np.log(pred.ravel()[idx]))
            lg.append(np.log(gt.ravel()[idx]))
    finally:
        head.cal_a.copy_(a0)
        head.cal_b.copy_(b0)
    if not lp:
        raise RuntimeError("No valid calibration pixels found")
    return np.concatenate(lp), np.concatenate(lg)


def calibrate_checkpoint(ckpt_path: str | Path, data_root: Path, split: str, imgsz: int, min_depth: float, max_depth: float,
                         max_images: int = 1000, device="cpu", seed: int = 42, report_path: Path | None = None) -> dict:
    """Fit (a, b) on ``split`` and write them into ``ckpt_path`` (in place). Returns the calibration record."""
    images = sorted(p for p in (data_root / "images" / split).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise FileNotFoundError(f"No images for calibration in {data_root / 'images' / split}")
    if len(images) > max_images:
        images = [images[i] for i in np.random.default_rng(seed).choice(len(images), max_images, replace=False)]
    model = load_model(ckpt_path, device, fuse=False)
    lp, lg = collect_calibration_pixels(model, images, data_root / "depth" / split, imgsz, min_depth, max_depth, seed=seed)
    a, b, mode = fit_log_affine(lp, lg)
    rmse_log_before = float(np.sqrt(np.mean((lp - lg) ** 2)))
    rmse_log_after = float(np.sqrt(np.mean((a * lp + b - lg) ** 2)))
    ckpt = load_ultralytics_ckpt(ckpt_path)
    head = depth_head(ckpt["model"])
    # checkpoints are stored in FP16; keep the two calibration scalars in FP32 so (a, b) are not rounded
    head.cal_a = torch.tensor([a], dtype=torch.float32)
    head.cal_b = torch.tensor([b], dtype=torch.float32)
    record = {"checkpoint": str(ckpt_path), "split": split, "num_images": len(images), "num_pixels": int(lp.size), "a": a, "b": b, "mode": mode,
              "fit_rmse_log_before": rmse_log_before, "fit_rmse_log_after": rmse_log_after}
    ckpt["log_affine_calibration"] = record
    save_ultralytics_ckpt(ckpt, ckpt_path)
    if report_path is not None:
        save_json(record, report_path)
    return record
