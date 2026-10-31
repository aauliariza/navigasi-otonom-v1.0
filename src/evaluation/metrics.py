"""Evaluation metrics of the proposal (Sub-bab 3.2.5).

Detection (Eq. 3.21-3.24): mAP50 / mAP50-95 come from the Ultralytics validator (single class, so mAP = AP);
Precision / Recall / F1 at the operating point are computed here by greedy one-to-one matching at IoU >= 0.5
of the detections kept at inference (conf >= 0.25).

Depth (Eq. 3.34-3.38): Abs_rel, Sq_rel, RMSE, RMSE_log, delta1..3 over valid pixels 0.001 <= D_gt <= 10 m,
predictions clamped to the same range. Pooled over all valid pixels of the split (N = number of valid pixels,
literal reading of Eq. 3.34) and, as a supplement, averaged per image.

Distance (Eq. 3.39-3.43): MAE, RMSE, MAPE over objects whose distance is valid. Navigation (Eq. 3.45): accuracy.
"""

from __future__ import annotations

import numpy as np

DEPTH_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3")


# ----------------------------------------------------------------------------------------------------------------- detection
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (Eq. 2.2) between xyxy boxes ``a`` (N, 4) and ``b`` (M, 4)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def match_detections(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> tuple[int, int, int]:
    """Greedy (highest score first) one-to-one matching. Returns (TP, FP, FN)."""
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    iou = box_iou(pred[np.argsort(-scores)], gt)
    used = np.zeros(len(gt), bool)
    tp = 0
    for row in iou:
        row = np.where(used, -1.0, row)
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            used[j] = True
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Eq. (3.22)-(3.24), with 0 when a denominator is 0."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def f1_score(p: float, r: float) -> float:
    """Harmonic mean (Eq. 3.24)."""
    return 2 * p * r / (p + r) if p + r else 0.0


# ----------------------------------------------------------------------------------------------------------------- depth
class DepthMetricAccumulator:
    """Accumulates Eq. (3.34)-(3.38) pooled over pixels and per image."""

    def __init__(self, min_depth: float = 0.001, max_depth: float = 10.0):
        self.min_depth, self.max_depth = min_depth, max_depth
        self.sums = dict.fromkeys(("abs_rel", "sq_rel", "se", "se_log", "d1", "d2", "d3"), 0.0)
        self.n = 0
        self.per_image: list[dict] = []

    def update(self, pred: np.ndarray, gt: np.ndarray, name: str = "") -> dict | None:
        """Add one image (pred and gt in metres, same shape). Returns that image's metrics (or None if no valid pixel)."""
        if pred.shape != gt.shape:
            raise ValueError(f"pred {pred.shape} and gt {gt.shape} must have the same shape")
        m = np.isfinite(gt) & (gt >= self.min_depth) & (gt <= self.max_depth)
        if not m.any():
            return None
        g = gt[m].astype(np.float64)
        p = np.nan_to_num(pred[m].astype(np.float64), nan=self.max_depth, posinf=self.max_depth, neginf=self.min_depth)
        p = np.clip(p, self.min_depth, self.max_depth)
        ratio = np.maximum(p / g, g / p)
        diff = p - g
        dlog = np.log(p) - np.log(g)
        s = {"abs_rel": np.sum(np.abs(diff) / g), "sq_rel": np.sum(diff**2 / g), "se": np.sum(diff**2), "se_log": np.sum(dlog**2),
             "d1": np.sum(ratio < 1.25), "d2": np.sum(ratio < 1.25**2), "d3": np.sum(ratio < 1.25**3)}
        for k, v in s.items():
            self.sums[k] += float(v)
        self.n += g.size
        img = self._finalize(s, g.size)
        img.update(image=name, valid_pixels=int(g.size))
        self.per_image.append(img)
        return img

    @staticmethod
    def _finalize(s: dict, n: int) -> dict[str, float]:
        return {"abs_rel": s["abs_rel"] / n, "sq_rel": s["sq_rel"] / n, "rmse": float(np.sqrt(s["se"] / n)),
                "rmse_log": float(np.sqrt(s["se_log"] / n)), "delta1": s["d1"] / n, "delta2": s["d2"] / n, "delta3": s["d3"] / n}

    def compute(self) -> dict[str, float]:
        """Pooled metrics (primary) plus ``<metric>_per_image_mean`` (supplementary)."""
        if self.n == 0:
            raise RuntimeError("No valid pixels were accumulated")
        out = {k: float(v) for k, v in self._finalize(self.sums, self.n).items()}
        for k in DEPTH_KEYS:
            out[f"{k}_per_image_mean"] = float(np.mean([r[k] for r in self.per_image]))
        out["valid_pixels"] = int(self.n)
        out["images"] = len(self.per_image)
        return out


# ----------------------------------------------------------------------------------------------------------------- distance / navigation
def distance_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """MAE, RMSE (m) and MAPE (%) -- Eq. (3.41)-(3.43). Inputs: valid objects only, gt > 0."""
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    if pred.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0}
    if np.any(gt <= 0):
        raise ValueError("Ground-truth distances must be > 0 for MAPE")
    err = np.abs(pred - gt)
    return {"mae": float(err.mean()), "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))), "mape": float(100.0 * np.mean(err / gt)), "n": int(pred.size)}


def navigation_accuracy(pred: list[str], gt: list[str]) -> dict[str, float]:
    """Eq. (3.45): percentage of scenarios whose decided direction equals the ground truth."""
    if len(pred) != len(gt):
        raise ValueError("pred and gt must have equal length")
    correct = sum(p == g for p, g in zip(pred, gt))
    return {"n_total": len(gt), "n_correct": correct, "accuracy_pct": 100.0 * correct / len(gt) if gt else float("nan")}
