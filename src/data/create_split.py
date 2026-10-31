"""Dataset split for every experiment: official SUN RGB-D train/test + scene-aware internal validation.

Official partition (``SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat``)::

    alltrain  5,285 samples  ->  train (fitting) + val (internal validation, carved out here)
    alltest   5,050 samples  ->  test  (final evaluation ONLY)

The proposal (Tabel 3.6) labels the 5,050 official test samples "validation". In this repository they are the
``test`` split and are never used for early stopping, best-checkpoint selection, hyperparameter choice,
log-affine calibration or EWM-SAW model selection (docs/EXPERIMENT_DESIGN_DECISIONS.md, keputusan S-1).

Internal validation (``val``) is drawn from ``alltrain`` by *groups* of samples that come from the same physical
recording, so near-duplicate views of one room cannot end up on both sides (data leakage). The group key is
derived from the portable scene identity (``<sensor>/<subset>/.../<sample>``) by :func:`scene_group_id`:

* ``xtion/sun3ddata/<location>/<sequence>/<frame>`` (SUN3D video sequences) -> ``xtion/sun3ddata/<location>``
  (every frame of every sequence recorded at that location is one group);
* names containing a capture-session stamp ``YYYY-MM-DD_hh-mm-ss_<serial>`` (``kv2/kinect2data``,
  ``kv2/align_kv2``, ``xtion/xtion_align_data``) -> ``<sensor>/<subset>/<stamp>``;
* everything else (``kv1/NYUdata``, ``kv1/b3dodata``, ``realsense/*``) -> the sample itself, because the folder
  names carry no recording identity. For NYUdata this is a known limitation (several NYU frames can come from the
  same room); an explicit ``group_overrides`` CSV (``scene_id,group``) can supply a finer mapping.

Groups are shuffled with a fixed seed independently per sensor and taken until ``fraction`` of that sensor's
samples is reached, so the validation set keeps the sensor mix of the training set. The result is written once to
``splits.json`` and re-used by every experiment (identical split for all 15 trainings).
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SESSION_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+)")


def scene_group_id(scene_id: str, overrides: dict[str, str] | None = None) -> str:
    """Recording-group key of a portable SUN RGB-D scene identity (see module docstring)."""
    if overrides and scene_id in overrides:
        return overrides[scene_id]
    parts = scene_id.split("/")
    if "sun3ddata" in parts:
        k = parts.index("sun3ddata")
        if len(parts) > k + 2:  # location folder exists below sun3ddata
            return "/".join(parts[: k + 2])
    m = SESSION_RE.search(parts[-1])
    if m and len(parts) >= 2:
        return "/".join(parts[:-1] + [m.group(1)])
    return scene_id


def load_group_overrides(path: str | Path | None) -> dict[str, str]:
    """Optional ``scene_id,group`` CSV."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"group_overrides file not found: {p}")
    with p.open(newline="", encoding="utf-8") as f:
        return {r["scene_id"]: r["group"] for r in csv.DictReader(f)}


def internal_validation_split(
    train_ids: list[str],
    fraction: float = 0.10,
    seed: int = 42,
    overrides: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Split official-train scene ids into ``train`` / ``val`` without sharing any recording group.

    Returns:
        dict with sorted ``train`` and ``val`` lists and ``groups`` (scene_id -> group key).
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"internal validation fraction must be in (0, 1), got {fraction}")
    if len(train_ids) < 2:
        raise ValueError("need at least two training samples to carve out a validation split")
    groups = {s: scene_group_id(s, overrides) for s in train_ids}
    by_sensor: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for s, g in groups.items():
        by_sensor[s.split("/")[0]][g].append(s)
    rng = np.random.default_rng(seed)
    val: set[str] = set()
    for sensor in sorted(by_sensor):
        gmap = by_sensor[sensor]
        keys = sorted(gmap)
        order = rng.permutation(len(keys))
        n_sensor = sum(len(v) for v in gmap.values())
        target = fraction * n_sensor
        taken = 0
        for idx in order:
            if taken >= target:
                break
            members = gmap[keys[idx]]
            # never move a whole sensor into val
            if taken + len(members) >= n_sensor:
                continue
            val.update(members)
            taken += len(members)
    if not val:  # tiny datasets: guarantee a non-empty validation split
        smallest = min({g for g in groups.values()}, key=lambda g: (sum(1 for x in groups.values() if x == g), g))
        val = {s for s, g in groups.items() if g == smallest}
    train = sorted(set(train_ids) - val)
    if not train:
        raise ValueError("internal validation consumed every training sample; lower the fraction")
    return {"train": train, "val": sorted(val), "groups": groups}


def check_group_leakage(split: dict[str, list[str]], groups: dict[str, str]) -> list[str]:
    """Group keys present in both ``train`` and ``val`` (must be empty)."""
    tg = {groups[s] for s in split["train"]}
    vg = {groups[s] for s in split["val"]}
    return sorted(tg & vg)
