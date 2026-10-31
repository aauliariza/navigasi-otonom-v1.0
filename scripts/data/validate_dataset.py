#!/usr/bin/env python
"""Entry point: ``python scripts/data/validate_dataset.py [--config configs/dataset/sunrgbd.yaml]``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.validate_dataset import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
