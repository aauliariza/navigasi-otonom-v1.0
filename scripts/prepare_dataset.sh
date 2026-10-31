#!/usr/bin/env bash
# Tahap 2-3: unduh (opsional), siapkan, dan validasi SUN RGB-D.
#   bash scripts/prepare_dataset.sh            # data mentah sudah ada di datasets/raw
#   bash scripts/prepare_dataset.sh --download # unduh dulu dari rgbd.cs.princeton.edu
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--download" ]]; then
  python -m src.data.download_dataset --out datasets/raw
fi
python -m src.data.prepare_sunrgbd --config configs/dataset/sunrgbd.yaml
python scripts/data/validate_dataset.py --config configs/dataset/sunrgbd.yaml
