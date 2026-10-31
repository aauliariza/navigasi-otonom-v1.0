"""Random seed and reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs.

    Args:
        seed (int): Seed value (proposal Tabel 3.7/3.8: 42).
        deterministic (bool): Request deterministic cuDNN kernels. Some CUDA ops have no deterministic
            implementation; ``warn_only=True`` keeps those runnable while logging a warning.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
