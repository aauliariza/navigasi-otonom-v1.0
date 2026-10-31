"""JSON / CSV / YAML writers used for every experiment artefact."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def _to_builtin(obj: Any) -> Any:
    """Convert NumPy / Path / torch scalars into JSON-serialisable Python objects."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item") and callable(obj.item) and getattr(obj, "ndim", 1) == 0:
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def save_json(data: Any, path: str | Path) -> Path:
    """Write ``data`` as indented UTF-8 JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(data), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_csv(rows: Iterable[dict], path: str | Path, fieldnames: list[str] | None = None) -> Path:
    """Write a list of dict rows to CSV; the header is the union of keys in first-seen order."""
    rows = [_to_builtin(r) for r in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            fieldnames.extend(k for k in r if k not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def load_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dict rows (values are strings)."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_yaml(data: Any, path: str | Path) -> Path:
    """Write ``data`` as YAML (keys keep insertion order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_to_builtin(data), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
