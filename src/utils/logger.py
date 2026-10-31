"""Console + file logging shared by every script."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to stdout and, optionally, to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve() for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FMT))
            logger.addHandler(fh)
    logger.propagate = False
    return logger
