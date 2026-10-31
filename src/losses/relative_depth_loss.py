"""Relative Depth Distillation, Eq. (2.13)-(2.14).

Ordinal label from the teacher (with tolerance ``tau``, proposal: 0.03)::

    r_ij = sign(log D_T(i) - log D_T(j))   if |log D_T(i) - log D_T(j)| > tau,  else 0

Ranking loss on the student log-depth difference ``dS_ij = log D_S(i) - log D_S(j)``::

    r_ij != 0 :  log(1 + exp(-r_ij * (dS_ij - m * r_ij)))         (Eq. 2.14, margin m)
    r_ij == 0 :  dS_ij^2                                           (equal-pair term)

Eq. (2.14) evaluated literally at ``r_ij = 0`` is the constant ``log 2`` (no gradient). The squared term for
"equal" pairs follows the ordinal ranking loss of Chen, Fu, Yang & Deng (2016), "Single-Image Depth Perception
in the Wild", NeurIPS 29 (arXiv:1604.03901), which Eq. (2.14) is derived from; ``equal_pair_loss: ignore``
reproduces the literal form. Pair sampling: ``num_pairs`` (proposal: 4,096) pairs per image, drawn uniformly
with replacement from valid pixels by a dedicated, seeded ``torch.Generator`` so the sampling sequence is
reproducible run to run.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PairSampler:
    """Seeded uniform sampler of pixel pairs among valid pixels (one generator per device)."""

    def __init__(self, num_pairs: int = 4096, seed: int = 42):
        if num_pairs < 1:
            raise ValueError("num_pairs must be >= 1")
        self.num_pairs = num_pairs
        self.seed = seed
        self._gens: dict[str, torch.Generator] = {}

    def __getstate__(self):
        """Generators are not picklable; they are re-created (re-seeded) after unpickling."""
        return {"num_pairs": self.num_pairs, "seed": self.seed, "_gens": {}}

    def generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._gens:
            g = torch.Generator(device=device)
            g.manual_seed(self.seed)
            self._gens[key] = g
        return self._gens[key]

    def __call__(self, valid_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = valid_idx.numel()
        g = self.generator(valid_idx.device)
        i = torch.randint(0, n, (self.num_pairs,), device=valid_idx.device, generator=g)
        j = torch.randint(0, n, (self.num_pairs,), device=valid_idx.device, generator=g)
        return valid_idx[i], valid_idx[j]


def ordinal_labels(dt: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Eq. (2.13) with tolerance: -1 / 0 / +1."""
    r = torch.sign(dt)
    r[dt.abs() <= tolerance] = 0
    return r


def relative_depth_loss(
    log_student: torch.Tensor,
    log_teacher: torch.Tensor,
    mask: torch.Tensor,
    sampler: PairSampler,
    tolerance: float = 0.03,
    margin: float = 0.0,
    equal_pair_loss: str = "squared",
) -> torch.Tensor:
    """Eq. (2.14) averaged over all sampled pairs of all images (``1/N_p``)."""
    if equal_pair_loss not in {"squared", "ignore"}:
        raise ValueError("equal_pair_loss must be 'squared' or 'ignore'")
    losses = []
    for b in range(log_student.shape[0]):
        valid_idx = mask[b].reshape(-1).nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() < 2:
            continue
        ia, ib = sampler(valid_idx)
        s, t = log_student[b].reshape(-1), log_teacher[b].reshape(-1).detach()
        ds, dt = s[ia] - s[ib], t[ia] - t[ib]
        r = ordinal_labels(dt, tolerance)
        ordered = r != 0
        pair = torch.zeros_like(ds)
        pair[ordered] = F.softplus(-r[ordered] * (ds[ordered] - margin * r[ordered]))
        if equal_pair_loss == "squared":
            pair[~ordered] = ds[~ordered].pow(2)
            losses.append(pair)
        else:
            losses.append(pair[ordered])
    if not losses or sum(x.numel() for x in losses) == 0:
        return (log_student * 0.0).sum()
    return torch.cat(losses).mean()
