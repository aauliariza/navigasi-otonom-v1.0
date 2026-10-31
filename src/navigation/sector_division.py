"""Five equal-width sectors of the global navigation ROI, Eq. (3.20): ``W_s = 0.60 W / 5 = 0.12 W``.

Order from left to right as listed in the proposal: FL (Front-Left), L (Left), CTR (Center), R (Right),
FR (Front-Right)."""

from __future__ import annotations

SECTOR_NAMES = ("FL", "L", "CTR", "R", "FR")


def divide_sectors(roi_xyxy, names=SECTOR_NAMES) -> list[dict]:
    """Return sector dicts ``{"name", "x1", "y1", "x2", "y2"}`` of equal width covering the ROI."""
    x1, y1, x2, y2 = roi_xyxy
    ws = (x2 - x1) / len(names)
    return [{"name": n, "x1": x1 + i * ws, "y1": y1, "x2": x1 + (i + 1) * ws, "y2": y2} for i, n in enumerate(names)]
