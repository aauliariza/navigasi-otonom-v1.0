#!/usr/bin/env bash
# Tahap 7 dan 11: evaluasi seluruh model deteksi dan depth (metrik + FPS + parameter + FLOPs + memori GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.evaluation.evaluate_detection --config configs/evaluation/detection.yaml --include-extra "$@"
python -m src.evaluation.evaluate_depth     --config configs/evaluation/depth.yaml     --include-extra "$@"
