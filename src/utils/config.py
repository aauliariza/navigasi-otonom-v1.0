"""Experiment configuration loading.

Every experiment is a YAML file. A file may declare ``_base_: <relative path>`` to inherit a shared
file (e.g. ``configs/detection/_common.yaml``); keys in the child override the base recursively.
This keeps the hyperparameters of compared experiments identical by construction (proposal Tabel 3.7/3.8).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving ``_base_`` inheritance chains."""
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    cfg.setdefault("_source_", str(path))
    return cfg


def resolve_path(p: str | Path | None) -> Path | None:
    """Resolve a path relative to the repository root (absolute paths are returned unchanged)."""
    if p is None:
        return None
    p = Path(p).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``key.sub=value`` CLI overrides (values parsed as YAML scalars)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got '{item}'")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg
