#!/usr/bin/env bash
# Evaluasi seluruh model deteksi dan depth (metrik + FPS + parameter + FLOPs + memori GPU).
#   1) split val  (validasi internal)  -> dipakai EWM-SAW (scripts/select_models.sh)
#   2) split test (official test 5.050) -> evaluasi akhir; TIDAK dipakai untuk memilih model
# Argumen tambahan diteruskan sebagai override, mis.: bash scripts/evaluate_all.sh device=0
set -euo pipefail
cd "$(dirname "$0")/.."
for SPLIT in val test; do
  python -m src.evaluation.evaluate_detection --config configs/evaluation/detection.yaml --include-extra --set split=$SPLIT "$@"
  python -m src.evaluation.evaluate_depth     --config configs/evaluation/depth.yaml     --include-extra --set split=$SPLIT "$@"
done
