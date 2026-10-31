"""SUN RGB-D depth decoding, metric conversion and valid-mask construction (proposal Sub-bab 3.2.3.2).

SUN RGB-D stores depth as uint16 PNG with a 3-bit circular rotation. Eq. (3.2) and (3.1):

    D_mm = (D_raw >> 3) | (D_raw << 13)        (uint16 arithmetic, i.e. a 16-bit rotate-right by 3)
    D_m  = D_mm / 1000

The same decode is used by the official SUNRGBDtoolbox (``read3dPoints.m``) and by the Ultralytics
``depth-sunrgbd.yaml`` converter. Depth is NEVER treated as an RGB image: it is read with
``cv2.IMREAD_UNCHANGED`` and kept as numeric values end to end.

Valid range (proposal Sub-bab 3.2.4.3): 0.001 <= D <= 10 m. Pixels outside the range, zero or non-finite
are invalid. By default out-of-range pixels are *invalidated* (set to 0, the invalid marker) instead of
being clipped to 10 m, because clipping would manufacture fake 10 m ground truth that then counts as
"valid" in the loss and in evaluation (see docs/EXPERIMENT_DESIGN_DECISIONS.md, keputusan D-3).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MIN_DEPTH_M = 0.001
MAX_DEPTH_M = 10.0
DEPTH_PNG_SCALE = 1000  # stored PNG value 1000 == 1 m (uint16 millimetres), 0 == invalid


def decode_sunrgbd_raw(raw: np.ndarray) -> np.ndarray:
    """Apply the SUN RGB-D bit rotation, Eq. (3.2): returns depth in millimetres (uint16)."""
    if raw.dtype != np.uint16:
        raise ValueError(f"SUN RGB-D depth must be uint16, got {raw.dtype} (was the PNG read as 8-bit?)")
    return np.bitwise_or(np.right_shift(raw, 3), np.left_shift(raw, 13)).astype(np.uint16)


def encode_sunrgbd_raw(depth_mm: np.ndarray) -> np.ndarray:
    """Inverse of :func:`decode_sunrgbd_raw` (rotate-left by 3). Used only to build synthetic test data."""
    d = depth_mm.astype(np.uint16)
    return np.bitwise_or(np.left_shift(d, 3), np.right_shift(d, 13)).astype(np.uint16)


def depth_mm_to_m(depth_mm: np.ndarray) -> np.ndarray:
    """Eq. (3.1): millimetres -> metres as float32."""
    return depth_mm.astype(np.float32) / 1000.0


def valid_depth_mask(depth_m: np.ndarray, min_depth: float = MIN_DEPTH_M, max_depth: float = MAX_DEPTH_M) -> np.ndarray:
    """Boolean mask of valid metric depth: finite and ``min_depth <= D <= max_depth``."""
    return np.isfinite(depth_m) & (depth_m >= min_depth) & (depth_m <= max_depth)


def apply_valid_range(
    depth_m: np.ndarray,
    min_depth: float = MIN_DEPTH_M,
    max_depth: float = MAX_DEPTH_M,
    out_of_range: str = "invalidate",
) -> np.ndarray:
    """Restrict depth to the valid range.

    Args:
        depth_m (np.ndarray): Metric depth (float32).
        out_of_range (str): ``invalidate`` (default) sets far pixels to 0; ``clip`` clamps them to ``max_depth``
            (reproduces the Ultralytics converter, provided only for comparison).
    """
    d = depth_m.astype(np.float32, copy=True)
    d[~np.isfinite(d)] = 0.0
    if out_of_range == "clip":
        far = d > max_depth
        d[far] = max_depth
    elif out_of_range == "invalidate":
        d[d > max_depth] = 0.0
    else:
        raise ValueError(f"out_of_range must be 'invalidate' or 'clip', got '{out_of_range}'")
    d[d < min_depth] = 0.0
    return d


def read_sunrgbd_depth(
    path: str | Path,
    min_depth: float = MIN_DEPTH_M,
    max_depth: float = MAX_DEPTH_M,
    out_of_range: str = "invalidate",
) -> np.ndarray:
    """Read a raw SUN RGB-D depth PNG and return metric depth (m) with invalid pixels set to 0."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"Unreadable depth file: {path}")
    if raw.ndim != 2:
        raise ValueError(f"Depth PNG must be single-channel, got shape {raw.shape}: {path}")
    return apply_valid_range(depth_mm_to_m(decode_sunrgbd_raw(raw)), min_depth, max_depth, out_of_range)


def save_metric_depth_png(path: str | Path, depth_m: np.ndarray, scale: int = DEPTH_PNG_SCALE) -> Path:
    """Save metric depth as uint16 PNG (``value = round(D_m * scale)``, 0 = invalid).

    This is the lossless numeric format consumed by the Ultralytics ``DepthDataset`` (``depth_scale: 1000``):
    1 mm resolution up to 65.535 m, so the 10 m SUN RGB-D range is represented exactly at sensor precision.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = np.nan_to_num(depth_m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    enc = np.rint(np.clip(d, 0, 65535 / scale) * scale).astype(np.uint16)
    if not cv2.imwrite(str(path), enc) or not path.is_file():
        raise OSError(f"Failed to write depth PNG: {path}")
    return path


def load_metric_depth_png(path: str | Path, scale: int = DEPTH_PNG_SCALE) -> np.ndarray:
    """Load a depth PNG written by :func:`save_metric_depth_png` back to metres (0 = invalid)."""
    enc = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if enc is None or enc.dtype != np.uint16 or enc.ndim != 2:
        raise OSError(f"Not a uint16 single-channel depth PNG: {path}")
    return enc.astype(np.float32) / float(scale)
