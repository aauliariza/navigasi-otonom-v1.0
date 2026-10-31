"""Checkpoint helpers: locating best/last weights and reading/writing Ultralytics checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils.config import resolve_path


def experiment_dir(cfg: dict) -> Path:
    """Directory of an experiment run: ``<output.root>/<output.group>/<experiment.id>``."""
    out = cfg.get("output", {})
    return resolve_path(out.get("root", "outputs")) / out.get("group", "misc") / cfg["experiment"]["id"]


def best_weights(cfg: dict) -> Path:
    """Path of the best checkpoint produced by an experiment config."""
    return experiment_dir(cfg) / "weights" / "best.pt"


def last_weights(cfg: dict) -> Path:
    """Path of the last checkpoint produced by an experiment config (used for resume)."""
    return experiment_dir(cfg) / "weights" / "last.pt"


def load_ultralytics_ckpt(path: str | Path) -> dict[str, Any]:
    """Load an Ultralytics ``.pt`` checkpoint dict (contains the pickled model object).

    ``weights_only=False`` is required because Ultralytics pickles the full ``nn.Module``; only load
    checkpoints you produced yourself.
    """
    return torch.load(str(path), map_location="cpu", weights_only=False)


def save_ultralytics_ckpt(ckpt: dict[str, Any], path: str | Path) -> Path:
    """Write an Ultralytics checkpoint dict back to disk."""
    path = Path(path)
    torch.save(ckpt, str(path))
    return path


def require_file(path: str | Path, what: str, hint: str) -> Path:
    """Raise a clear error when a required artefact (teacher checkpoint, report, ...) is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}\n  {hint}")
    return path
