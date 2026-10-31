"""Object distance from the metric depth inside the inner ROI, proposal Sub-bab 3.2.4.6 (Eq. 3.18-3.19).

Procedure for one object (identical for predicted depth and ground-truth depth):
1. depth values of the inner ROI (``N_total`` pixels);
2. valid filtering: finite, non-zero and inside [0.1, 10] m;
3. ``R_valid = N_valid / N_total`` (Eq. 3.19); the distance is VALID only if ``R_valid >= 0.30``;
4. percentile clipping at the 5th / 95th percentile of the valid values (``clip`` = clamp, default; ``trim`` =
   discard values outside the band);
5. ``D_obs = median(Z_clip)`` (Eq. 3.18).

Mode ``axial`` (primary): Z along the optical axis, i.e. the depth value itself. Mode ``euclidean`` (optional,
needs camera intrinsics at depth resolution): every pixel's depth is converted to its ray length
``Z * sqrt(((u - cx)/fx)^2 + ((v - cy)/fy)^2 + 1)`` before steps 4-5.
The mean of the whole depth map or a single pixel is never used.
"""

from __future__ import annotations

import numpy as np

from src.integration.inner_roi import inner_roi


def estimate_distance(
    depth: np.ndarray,
    box_depth_xyxy,
    roi_ratio: float = 0.6,
    roi_min_size: int = 4,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    min_valid_ratio: float = 0.30,
    percentiles: tuple[float, float] = (5.0, 95.0),
    clip_mode: str = "clip",
    mode: str = "axial",
    intrinsics: np.ndarray | None = None,
) -> dict:
    """Distance record of one object. ``box_depth_xyxy`` must already be in depth-map coordinates."""
    H, W = depth.shape[:2]
    c0, r0, c1, r1 = inner_roi(box_depth_xyxy, roi_ratio, roi_min_size, (H, W))
    roi = depth[r0:r1, c0:c1].astype(np.float64)
    n_total = int(roi.size)
    rec = {"roi": [c0, r0, c1, r1], "n_total": n_total, "n_valid": 0, "valid_ratio": 0.0, "distance_m": None, "valid": False, "reason": ""}
    if n_total == 0:
        rec["reason"] = "empty_roi"
        return rec
    valid = np.isfinite(roi) & (roi != 0) & (roi >= min_depth) & (roi <= max_depth)
    rec["n_valid"] = int(valid.sum())
    rec["valid_ratio"] = rec["n_valid"] / n_total
    if rec["valid_ratio"] < min_valid_ratio or rec["n_valid"] == 0:
        rec["reason"] = "low_valid_ratio"
        return rec
    z = roi[valid]
    if mode == "euclidean":
        if intrinsics is None:
            raise ValueError("euclidean mode needs camera intrinsics (at depth-map resolution)")
        vv, uu = np.nonzero(valid)
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        u, v = uu + c0 + 0.5, vv + r0 + 0.5
        z = z * np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)
    elif mode != "axial":
        raise ValueError(f"mode must be 'axial' or 'euclidean', got '{mode}'")
    lo, hi = np.percentile(z, percentiles)
    if clip_mode == "clip":
        z = np.clip(z, lo, hi)
    elif clip_mode == "trim":
        z = z[(z >= lo) & (z <= hi)]
    else:
        raise ValueError("clip_mode must be 'clip' or 'trim'")
    rec.update(distance_m=float(np.median(z)), valid=True, reason="ok", p_low=float(lo), p_high=float(hi))
    return rec
