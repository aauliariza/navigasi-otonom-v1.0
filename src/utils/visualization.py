"""Drawing helpers: depth colormaps, boxes with distance labels, navigation sector overlay."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

STATUS_COLORS = {"free": (60, 180, 75), "blocked": (40, 40, 220)}  # BGR


def colorize_depth(depth_m: np.ndarray, max_depth: float = 10.0, invalid_color=(0, 0, 0)) -> np.ndarray:
    """Turbo colormap of metric depth for *display only* (never used as training data)."""
    d = np.nan_to_num(depth_m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = d > 0
    norm = np.clip(d / max_depth, 0, 1)
    vis = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = invalid_color
    return vis


def draw_boxes(
    img: np.ndarray,
    boxes_xyxy: Iterable,
    labels: Iterable[str] | None = None,
    colors: Iterable[tuple[int, int, int]] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw xyxy boxes with optional text labels on a copy of ``img`` (BGR)."""
    out = img.copy()
    boxes = list(boxes_xyxy)
    labels = list(labels) if labels is not None else [""] * len(boxes)
    colors = list(colors) if colors is not None else [(0, 200, 255)] * len(boxes)
    for b, text, c in zip(boxes, labels, colors):
        x1, y1, x2, y2 = (int(round(v)) for v in b[:4])
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness)
        if text:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(y1, th + 4)
            cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty), c, -1)
            cv2.putText(out, text, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_sectors(img: np.ndarray, sectors: list[dict], decision: str | None = None, alpha: float = 0.25) -> np.ndarray:
    """Overlay the five navigation sectors (green=free, red=blocked) and the chosen direction."""
    out = img.copy()
    overlay = img.copy()
    h = img.shape[0]
    for s in sectors:
        x1, x2 = int(round(s["x1"])), int(round(s["x2"]))
        color = STATUS_COLORS["blocked" if s["blocked"] else "free"]
        cv2.rectangle(overlay, (x1, 0), (x2, h - 1), color, -1)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    for s in sectors:
        x1, x2 = int(round(s["x1"])), int(round(s["x2"]))
        cv2.rectangle(out, (x1, 0), (x2, h - 1), (255, 255, 255), 1)
        txt = f"{s['name']}:{'B' if s['blocked'] else 'F'}"
        cv2.putText(out, txt, (x1 + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, txt, (x1 + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if decision:
        cv2.putText(out, decision, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, decision, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def side_by_side(*images: np.ndarray, height: int | None = None) -> np.ndarray:
    """Concatenate images horizontally after resizing them to a common height."""
    height = height or images[0].shape[0]
    resized = [cv2.resize(im, (int(im.shape[1] * height / im.shape[0]), height)) for im in images]
    return np.concatenate(resized, axis=1)
