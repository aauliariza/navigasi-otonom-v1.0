"""Spatial alignment of detection boxes to the depth-map grid, proposal Eq. (3.11)-(3.12).

``x' = x * W_D / W_I``,  ``y' = y * H_D / H_I``

Both models receive the same RGB frame. YOLO26 returns boxes in original-image pixels (its letterbox is undone
by Ultralytics), while YOLO26-Depth sees the frame stretch-resized to imgsz x imgsz and returns a map at 1/4 of
that size. A stretch resize is a pure per-axis scaling of the full frame, so Eq. (3.11)-(3.12) map image
coordinates to depth coordinates exactly (no offset, no padding to compensate).
"""

from __future__ import annotations

import numpy as np


def scale_boxes_to_depth(boxes_xyxy: np.ndarray, image_hw: tuple[int, int], depth_hw: tuple[int, int]) -> np.ndarray:
    """Map (N, 4) xyxy boxes from image coordinates to depth-map coordinates and clip to the depth map."""
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4).copy()
    hi, wi = image_hw
    hd, wd = depth_hw
    b[:, [0, 2]] *= wd / wi
    b[:, [1, 3]] *= hd / hi
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, wd)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, hd)
    return b


def scale_intrinsics(K: np.ndarray, image_hw: tuple[int, int], depth_hw: tuple[int, int]) -> np.ndarray:
    """Scale a 3x3 camera matrix from image resolution to depth-map resolution (same per-axis factors)."""
    K = np.asarray(K, dtype=np.float64).copy()
    sx, sy = depth_hw[1] / image_hw[1], depth_hw[0] / image_hw[0]
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K
