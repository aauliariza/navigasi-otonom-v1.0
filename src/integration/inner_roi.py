"""Inner ROI of a bounding box, proposal Eq. (3.13)-(3.17).

For ``B = (x1, y1, x2, y2)``, ``w = x2 - x1``, ``h = y2 - y1`` and margin ``m = (1 - ratio) / 2`` (ratio 0.6 ->
m = 0.2 on every side)::

    x1' = x1 + m w,  x2' = x2 - m w,  y1' = y1 + m h,  y2' = y2 - m h

(Eq. 3.17 in the proposal reads ``y2' = y1 - 0.2h``; that is a typo -- it would place y2' above y1 -- and the
symmetric form ``y2 - 0.2h`` stated in the text, "60% bagian tengah", is implemented.) Very small ROIs are grown
around their centre to at least ``min_size`` pixels per side (proposal: 4 px), never beyond the map.
The result is an integer half-open pixel window ``[r0, r1) x [c0, c1)`` on the depth map.
"""

from __future__ import annotations

import math


def inner_roi(box_xyxy, ratio: float = 0.6, min_size: int = 4, bounds_hw: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    """Return integer ``(c0, r0, c1, r1)`` of the inner ROI (columns = x, rows = y)."""
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    w, h = max(x2 - x1, 0.0), max(y2 - y1, 0.0)
    m = (1.0 - ratio) / 2.0
    rx1, rx2, ry1, ry2 = x1 + m * w, x2 - m * w, y1 + m * h, y2 - m * h

    def grow(a, b, limit):
        if b - a < min_size:
            c = (a + b) / 2.0
            a, b = c - min_size / 2.0, c + min_size / 2.0
        if limit is not None:
            if a < 0:
                a, b = 0.0, min(float(limit), b - a)
            if b > limit:
                a, b = max(0.0, a - (b - limit)), float(limit)
        return a, b

    H, W = bounds_hw if bounds_hw is not None else (None, None)
    rx1, rx2 = grow(rx1, rx2, W)
    ry1, ry2 = grow(ry1, ry2, H)
    c0, c1 = int(math.floor(rx1)), int(math.ceil(rx2))
    r0, r1 = int(math.floor(ry1)), int(math.ceil(ry2))
    if W is not None:
        c0, c1 = max(0, c0), min(W, c1)
    if H is not None:
        r0, r1 = max(0, r0), min(H, r1)
    return c0, r0, c1, r1
