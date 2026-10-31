"""Download the official SUN RGB-D files (https://rgbd.cs.princeton.edu/).

Files (all from the official website):
* ``SUNRGBD.zip``             RGB, raw/refined depth, intrinsics, 2D polygon annotations (~6.4 GB)
* ``SUNRGBDtoolbox.zip``      contains ``traintestSUNRGBD/allsplit.mat`` (official 5,285 / 5,050 split)
* ``SUNRGBDMeta2DBB_v2.mat``  2D bounding boxes

Usage::

    python -m src.data.download_dataset --out datasets/raw

If the server is unreachable (e.g. a firewall), download the three files manually with a browser and
place/extract them under ``datasets/raw`` so that ``datasets/raw/SUNRGBD/``, ``datasets/raw/SUNRGBDtoolbox/``
and ``datasets/raw/SUNRGBDMeta2DBB_v2.mat`` exist.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from src.utils.config import resolve_path
from src.utils.logger import get_logger

LOG = get_logger("download_dataset")
BASE = "https://rgbd.cs.princeton.edu/data/"
FILES = {"SUNRGBD.zip": True, "SUNRGBDtoolbox.zip": True, "SUNRGBDMeta2DBB_v2.mat": False}  # name -> extract


def download(url: str, dest: Path) -> Path:
    """Stream ``url`` to ``dest`` (skips when the file already exists)."""
    if dest.is_file() and dest.stat().st_size > 0:
        LOG.info("Exists, skipping: %s", dest)
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    LOG.info("Downloading %s -> %s", url, dest)
    with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Download official SUN RGB-D files")
    ap.add_argument("--out", default="datasets/raw")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args(argv)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, extract in FILES.items():
        path = download(BASE + name, out / name)
        if extract:
            LOG.info("Extracting %s", path)
            with zipfile.ZipFile(path) as z:
                z.extractall(out)
            if not args.keep_zip:
                path.unlink()
    LOG.info("Done. Next: python -m src.data.prepare_sunrgbd --config configs/dataset/sunrgbd.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
