"""Entropy Weight Method (EWM), proposal Eq. (3.29)-(3.32).

Given the decision matrix ``X`` (m alternatives x n criteria, non-negative raw values):

    p_ij = x_ij / sum_i x_ij                        (3.29)
    e_j  = -1/ln(m) * sum_i p_ij ln p_ij            (3.30)   with 0 * ln 0 := 0
    d_j  = 1 - e_j                                   (3.31)
    w_j  = d_j / sum_j d_j                           (3.32)

Edge cases (handled explicitly, never silently producing NaN):
* a criterion whose column sums to 0 -> p_ij = 1/m (no information) -> e_j = 1, d_j = 0;
* a criterion with identical values for every alternative -> p_ij = 1/m -> e_j = 1, d_j = 0 (zero weight: it
  cannot discriminate between alternatives);
* every criterion non-discriminating (sum d_j = 0) -> equal weights 1/n, flagged in the result;
* m = 1 -> ln(m) = 0: EWM is undefined, a ValueError is raised;
* negative values are rejected (EWM proportions require x >= 0).

The proposal applies Eq. (3.29) to the raw values ``x_ij`` (``basis='raw'``, default). Applying it to the
SAW-normalised matrix ``r_ij`` is available as ``basis='normalized'`` for sensitivity analysis.
"""

from __future__ import annotations

import numpy as np


def entropy_weights(X: np.ndarray) -> dict:
    """Return dict with ``p``, ``entropy``, ``diversification``, ``weights`` and ``equal_weight_fallback``."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Decision matrix must be 2-D (alternatives x criteria)")
    m, n = X.shape
    if m < 2:
        raise ValueError("EWM needs at least two alternatives (ln(m) = 0 for m = 1)")
    if not np.isfinite(X).all():
        raise ValueError("Decision matrix contains NaN/Inf")
    if (X < 0).any():
        raise ValueError("EWM requires non-negative criterion values")
    col = X.sum(axis=0)
    P = np.where(col > 0, X / np.where(col > 0, col, 1.0), 1.0 / m)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(P > 0, P * np.log(P), 0.0)
    e = -plogp.sum(axis=0) / np.log(m)
    e = np.clip(e, 0.0, 1.0)  # remove tiny floating-point excursions
    d = 1.0 - e
    fallback = bool(d.sum() <= 1e-15)
    w = np.full(n, 1.0 / n) if fallback else d / d.sum()
    return {"p": P, "entropy": e, "diversification": d, "weights": w, "equal_weight_fallback": fallback}
