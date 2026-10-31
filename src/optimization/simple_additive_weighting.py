"""Simple Additive Weighting (SAW), proposal Eq. (3.27), (3.28), (3.33).

Normalisation:
* benefit criterion:  r_ij = x_ij / max_i x_ij                            (3.27)
* cost criterion:     r_ij = min_i x_ij / x_ij

The proposal prints Eq. (3.28) as ``x_ij / min(x_ij)``; that expression is >= 1 and *grows* with the cost, so the
worst alternative would score highest. The standard SAW cost normalisation ``min / x`` (Fishburn, 1967;
MacCrimmon, 1968; used by Linh, Son & Thao, 2025, ETASR 15(1), doi:10.48084/etasr.9518) is implemented instead
-- see docs/EXPERIMENT_DESIGN_DECISIONS.md, keputusan M-2.

Zero handling: benefit with max = 0 -> r = 0; cost with x = 0 -> r = 1 for the zero-cost alternatives
(they are the best possible) and min/x = 0 for the others.

Score: V_i = sum_j w_j r_ij (3.33); the ranking sorts V descending (ties keep input order).
"""

from __future__ import annotations

import numpy as np


def saw_normalize(X: np.ndarray, benefit: list[bool]) -> np.ndarray:
    """Column-wise SAW normalisation."""
    X = np.asarray(X, dtype=np.float64)
    R = np.zeros_like(X)
    for j, is_benefit in enumerate(benefit):
        col = X[:, j]
        if is_benefit:
            mx = col.max()
            R[:, j] = col / mx if mx > 0 else 0.0
        else:
            mn = col.min()
            if mn > 0:
                R[:, j] = mn / col
            else:
                R[:, j] = np.where(col == 0, 1.0, 0.0)
    return R


def saw_scores(R: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Eq. (3.33)."""
    return np.asarray(R, dtype=np.float64) @ np.asarray(weights, dtype=np.float64)


def rank(scores: np.ndarray) -> np.ndarray:
    """1-based ranks (1 = highest score); stable for ties."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks
