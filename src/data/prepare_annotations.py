"""SUN RGB-D 2D bounding boxes -> single-class YOLO labels (proposal Sub-bab 3.2.3.3, Eq. 3.3-3.8).

Source (default): ``SUNRGBDMeta2DBB_v2.mat`` from the SUN RGB-D website. Field layout, as used by the
reference converters crmauceri/SUNRGBD_COCO (``SUNRGB_to_COCO.m``) and facebookresearch/votenet
(``extract_rgbd_data_v2.m``)::

    SUNRGBDMeta2DBB(i).sequenceName           scene path
    SUNRGBDMeta2DBB(i).groundtruth2DBB(k).gtBb2D     [x, y, w, h] in pixels
    SUNRGBDMeta2DBB(i).groundtruth2DBB(k).classname  object label (discarded: single class)

Alternative source: the per-scene ``annotation2Dfinal/index.json`` polygons (bounding box = polygon extent).

Cleaning rules:
* non-finite or non-positive w/h -> ``invalid``
* box entirely outside the image -> ``outside``; partially outside -> clipped to the image (``clipped``)
* side < ``min_side`` (4 px) after clipping -> ``too_small``

Every kept box becomes class 0 ("objek"); YOLO line: ``0 x_c' y_c' w' h'`` with Eq. (3.3)-(3.8).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.pair_rgb_depth import scene_relpath

CLASS_ID = 0
CLASS_NAME = "objek"


def _cell_list(x) -> list:
    """Flatten a scipy-loaded MATLAB struct array / cell array into a Python list."""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return list(x.ravel())
    return [x]


def load_meta2dbb(mat_path: str | Path) -> dict[str, list[list[float]]]:
    """Read ``SUNRGBDMeta2DBB_v2.mat`` -> {scene_id: [[x, y, w, h], ...]}."""
    from scipy.io import loadmat

    mat_path = Path(mat_path)
    if not mat_path.is_file():
        raise FileNotFoundError(
            f"{mat_path} not found. Download https://rgbd.cs.princeton.edu/data/SUNRGBDMeta2DBB_v2.mat"
        )
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "SUNRGBDMeta2DBB" not in mat:
        raise KeyError(f"{mat_path} has no 'SUNRGBDMeta2DBB'; found {[k for k in mat if not k.startswith('__')]}")
    out: dict[str, list[list[float]]] = {}
    for entry in _cell_list(mat["SUNRGBDMeta2DBB"]):
        scene = scene_relpath(str(getattr(entry, "sequenceName", "")))
        boxes = []
        for b in _cell_list(getattr(entry, "groundtruth2DBB", None)):
            gt = np.asarray(getattr(b, "gtBb2D", []), dtype=np.float64).ravel()
            boxes.append(gt.tolist() if gt.size == 4 else [np.nan] * 4)
        out.setdefault(scene, []).extend(boxes)
    return out


def load_annotation2d(scene_dir: str | Path) -> list[list[float]]:
    """Read ``annotation2Dfinal/index.json`` polygons of one scene -> [[x, y, w, h], ...]."""
    f = Path(scene_dir) / "annotation2Dfinal" / "index.json"
    if not f.is_file():
        return []
    data = json.loads(f.read_text(encoding="utf-8"))
    boxes = []
    for frame in data.get("frames") or []:
        for poly in (frame or {}).get("polygon") or []:
            xs = np.asarray((poly or {}).get("x") or [], dtype=np.float64)
            ys = np.asarray((poly or {}).get("y") or [], dtype=np.float64)
            if xs.size < 2 or ys.size < 2:
                boxes.append([np.nan] * 4)
                continue
            x0, y0 = xs.min(), ys.min()
            boxes.append([x0, y0, xs.max() - x0, ys.max() - y0])
    return boxes


def clean_box(box_xywh, img_w: int, img_h: int, min_side: float = 4.0) -> tuple[list[float] | None, str]:
    """Validate/clip one ``[x, y, w, h]`` box. Returns (xyxy or None, status)."""
    x, y, w, h = (float(v) for v in box_xywh)
    if not all(np.isfinite([x, y, w, h])) or w <= 0 or h <= 0:
        return None, "invalid"
    x1, y1, x2, y2 = x, y, x + w, y + h
    if x2 <= 0 or y2 <= 0 or x1 >= img_w or y1 >= img_h:
        return None, "outside"
    cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(float(img_w), x2), min(float(img_h), y2)
    status = "clipped" if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2) else "ok"
    if (cx2 - cx1) < min_side or (cy2 - cy1) < min_side:
        return None, "too_small"
    return [cx1, cy1, cx2, cy2], status


def xyxy_to_yolo(xyxy, img_w: int, img_h: int) -> list[float]:
    """Eq. (3.3)-(3.8): centre + size normalised by the image size."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    xc, yc = x1 + w / 2.0, y1 + h / 2.0
    return [xc / img_w, yc / img_h, w / img_w, h / img_h]


def convert_boxes(boxes_xywh, img_w: int, img_h: int, min_side: float = 4.0) -> tuple[list[list[float]], Counter]:
    """Clean every box of an image and return (yolo rows, status counter)."""
    rows, stats = [], Counter()
    for b in boxes_xywh:
        xyxy, status = clean_box(b, img_w, img_h, min_side)
        stats[status] += 1
        if xyxy is not None:
            rows.append(xyxy_to_yolo(xyxy, img_w, img_h))
    return rows, stats


def write_yolo_label(path: str | Path, rows: list[list[float]]) -> Path:
    """Write a YOLO label file (an empty file means "image with no objects")."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{CLASS_ID} {r[0]:.6f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f}\n" for r in rows), encoding="utf-8")
    return path


def read_yolo_label(path: str | Path) -> np.ndarray:
    """Read a YOLO label file into an (N, 5) float array."""
    p = Path(path)
    if not p.is_file() or not p.read_text().strip():
        return np.zeros((0, 5), dtype=np.float64)
    return np.loadtxt(p, ndmin=2, dtype=np.float64)


def yolo_to_xyxy(rows: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Inverse conversion (normalised YOLO -> pixel xyxy), used by validation, GT distance and tests."""
    if rows.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    xc, yc, w, h = rows[:, 1] * img_w, rows[:, 2] * img_h, rows[:, 3] * img_w, rows[:, 4] * img_h
    return np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)
