"""Shared pytest fixtures (tiny models, synthetic dataset)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models  # noqa: E402,F401


def _args(imgsz=64, **kw):
    from ultralytics.cfg import get_cfg

    a = get_cfg()
    a.imgsz = imgsz
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.fixture
def make_args():
    return _args


@pytest.fixture
def det_pair():
    """(teacher, student) tiny YOLO26n detection models with nc=1 and training args attached."""
    from models.common.model_utils import attention_yaml, build_from_yaml

    torch.manual_seed(0)
    teacher = build_from_yaml(attention_yaml("yolo26n.yaml", "ca"), "detect", nc=1)
    student = build_from_yaml("yolo26n.yaml", "detect", nc=1)
    for m in (teacher, student):
        m.args = _args()
    return teacher, student


@pytest.fixture
def depth_pair():
    """(teacher, student) tiny YOLO26n-Depth models (teacher with CBAM)."""
    from models.common.model_utils import attention_yaml, build_from_yaml

    torch.manual_seed(0)
    teacher = build_from_yaml(attention_yaml("yolo26n-depth.yaml", "cbam", task="depth"), "depth")
    student = build_from_yaml("yolo26n-depth.yaml", "depth")
    for m in (teacher, student):
        m.args = _args()
    return teacher, student


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    """Prepared + validated synthetic dataset (raw SUN RGB-D layout -> YOLO layout)."""
    from scripts_helpers import build_synthetic_dataset

    return build_synthetic_dataset(tmp_path_factory.mktemp("synthetic"))
