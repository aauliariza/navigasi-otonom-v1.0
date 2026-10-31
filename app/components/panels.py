"""Formatting helpers that turn a pipeline result into the interface's tables and text."""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.visualization import colorize_depth, draw_boxes, draw_sectors

DIRECTION_TEXT = {
    "FORWARD": "FORWARD - sektor CTR bebas",
    "TURN_LEFT": "TURN_LEFT - sektor L/FL bebas",
    "TURN_RIGHT": "TURN_RIGHT - sektor R/FR bebas",
    "STOP": "STOP - seluruh sektor terhalang",
}


def detection_panel(bgr: np.ndarray, result: dict) -> np.ndarray:
    """RGB (for display) with boxes, confidence + distance labels, the five sectors and the decision."""
    labels, colors = [], []
    for o in result["objects"]:
        labels.append(f"#{o['id']} {o['confidence']:.2f} | " + (f"{o['distance_m']:.2f} m" if o["valid"] else "invalid"))
        colors.append((0, 200, 255) if o["valid"] else (160, 160, 160))
    vis = draw_sectors(draw_boxes(bgr, [o["bbox"] for o in result["objects"]], labels, colors), result["sectors"], result["direction"])
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def depth_panel(result: dict, image_hw: tuple[int, int], max_depth: float = 10.0) -> np.ndarray:
    """Depth colormap resized to the image size (display only)."""
    vis = colorize_depth(result["depth"], max_depth)
    vis = cv2.resize(vis, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def objects_table(result: dict) -> list[list]:
    """Rows: id, bbox, confidence, distance (m), valid ratio, status."""
    return [[o["id"], ", ".join(f"{v:.0f}" for v in o["bbox"]), round(o["confidence"], 3),
             round(o["distance_m"], 3) if o["valid"] else None, round(o["valid_ratio"], 3), "valid" if o["valid"] else f"invalid ({o['reason']})"]
            for o in result["objects"]]


def sectors_table(result: dict) -> list[list]:
    """Rows: sector, status, blocking objects."""
    return [[s["name"], "BLOCKED" if s["blocked"] else "FREE", "; ".join(f"#{b['index']} ({b['reason']})" for b in s["blockers"])]
            for s in result["sectors"]]


def decision_text(result: dict) -> str:
    """Human readable decision + timing."""
    t = result["timing_ms"]
    return f"{DIRECTION_TEXT[result['direction']]}\nWaktu: deteksi {t['detection']:.1f} ms, depth {t['depth']:.1f} ms, fusion+navigasi {t['fusion_navigation']:.1f} ms, total {t['total']:.1f} ms"
