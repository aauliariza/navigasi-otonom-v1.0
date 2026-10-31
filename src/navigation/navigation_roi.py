"""Global navigation ROI, proposal Sub-bab 3.2.4.7: the central 60% of the frame width, full frame height,
i.e. ``0.20 W <= x < 0.80 W``."""

from __future__ import annotations


def global_navigation_roi(width: int, height: int, width_ratio: float = 0.6) -> tuple[float, float, float, float]:
    """Return (x1, y1, x2, y2) of the global navigation ROI."""
    if not 0 < width_ratio <= 1:
        raise ValueError("width_ratio must be in (0, 1]")
    margin = (1.0 - width_ratio) / 2.0 * width
    return margin, 0.0, width - margin, float(height)
