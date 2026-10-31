#!/usr/bin/env bash
# Tahap 12-13: EWM + SAW untuk deteksi dan depth, lalu pemilihan student KD untuk integrasi.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.optimization.model_selection --config configs/evaluation/selection.yaml "$@"
