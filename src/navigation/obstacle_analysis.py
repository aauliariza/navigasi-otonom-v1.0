"""Sector occupancy and blocking, proposal Sub-bab 3.2.4.7 / 3.2.5.4 (Eq. 3.44).

Occupancy: an object occupies a sector when ``Area(B ∩ S) / Area(B) >= min_overlap`` (0.05). Objects that do
not intersect the global navigation ROI do not influence the decision.

Blocking: a sector is ``blocked`` when at least one occupying object satisfies any ACTIVE condition:
* ``distance < block_distance`` (1.0 m), for a valid distance;
* the distance is invalid and ``invalid_depth_blocks`` is on;
* the fusion confidence is below ``low_conf_threshold`` (0.20) and ``low_confidence_blocks`` is on.
Otherwise the sector is ``free``. Distances are NOT recomputed here; the fusion output is used as is.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BlockingRules:
    """Configurable blocking conditions (defaults = proposal)."""

    min_overlap: float = 0.05
    block_distance: float = 1.0
    invalid_depth_blocks: bool = True
    low_confidence_blocks: bool = True
    low_conf_threshold: float = 0.20


def overlap_ratio(box, sector) -> float:
    """Eq. (3.44): Area(B ∩ S) / Area(B)."""
    x1, y1, x2, y2 = box
    area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    if area <= 0:
        return 0.0
    iw = max(0.0, min(x2, sector["x2"]) - max(x1, sector["x1"]))
    ih = max(0.0, min(y2, sector["y2"]) - max(y1, sector["y1"]))
    return iw * ih / area


def blocking_reason(obj: dict, rules: BlockingRules) -> str | None:
    """Why an object blocks (``near`` / ``invalid_depth`` / ``low_confidence``) or None."""
    if obj.get("valid") and obj.get("distance_m") is not None and obj["distance_m"] < rules.block_distance:
        return "near"
    if not obj.get("valid") and rules.invalid_depth_blocks:
        return "invalid_depth"
    if rules.low_confidence_blocks and obj.get("fusion_confidence", obj.get("confidence", 1.0)) < rules.low_conf_threshold:
        return "low_confidence"
    return None


def analyse_sectors(objects: list[dict], sectors: list[dict], roi_xyxy, rules: BlockingRules) -> list[dict]:
    """Return a copy of ``sectors`` with ``occupants``, ``blockers`` (index, reason) and ``blocked`` flags.

    Args:
        objects: dicts with ``bbox`` (image xyxy), ``valid``, ``distance_m``, ``confidence``, ``fusion_confidence``.
    """
    out = []
    for s in sectors:
        occ, blk = [], []
        for i, o in enumerate(objects):
            if overlap_ratio(o["bbox"], {"x1": roi_xyxy[0], "y1": roi_xyxy[1], "x2": roi_xyxy[2], "y2": roi_xyxy[3]}) <= 0:
                continue
            r = overlap_ratio(o["bbox"], s)
            if r >= rules.min_overlap:
                occ.append({"index": i, "overlap": r})
                reason = blocking_reason(o, rules)
                if reason:
                    blk.append({"index": i, "reason": reason})
        out.append({**s, "occupants": occ, "blockers": blk, "blocked": bool(blk)})
    return out
