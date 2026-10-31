"""YOLO26 detection model builders (YOLO26x teacher, YOLO26n student, optional CA teacher).

Model definitions are the official Ultralytics YAMLs (``yolo26{n,x}.yaml``); the only architectural change in
this repository is the optional Coordinate Attention variant of the teacher (:func:`model_yaml_for`), which
inserts three ``CoordAtt`` layers on P3/P4/P5 before the Detect head (see models/common/model_utils.py).
The student is never modified.
"""

from __future__ import annotations

from pathlib import Path

from models.common.model_utils import attention_yaml, base_yaml
from src.utils.io import save_yaml


def model_yaml_for(model_cfg: dict, task: str, out_dir: str | Path, nc: int = 1) -> str:
    """Return the YAML to train from: the official name, or a generated attention YAML written to ``out_dir``.

    Args:
        model_cfg (dict): ``{"arch": "yolo26x.yaml", "attention": "none"|"ca"|"cbam", "attention_args": {...}}``.
        task (str): ``detect`` or ``depth``.
        out_dir (Path): Where generated YAMLs are stored (inside the experiment folder, for traceability).
    """
    attention = str(model_cfg.get("attention", "none")).lower()
    if attention in {"none", "", "null"}:
        cfg = base_yaml(model_cfg["arch"])
        cfg["nc"] = nc
        path = Path(out_dir) / f"model_{Path(model_cfg['arch']).stem}.yaml"
    else:
        cfg = attention_yaml(model_cfg["arch"], attention, model_cfg.get("attention_args"), task=task, nc=nc)
        path = Path(out_dir) / f"model_{Path(model_cfg['arch']).stem}_{attention}.yaml"
    cfg.pop("yaml_file", None)
    save_yaml(cfg, path)
    return str(path)
