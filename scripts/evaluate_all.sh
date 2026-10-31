#!/usr/bin/env bash
# Evaluasi seluruh model deteksi dan depth (metrik + FPS + parameter + FLOPs + memori GPU).
#   split val  -> selalu (mode proposal: alltest 5.050; mode internal: validasi internal) -> dipakai EWM-SAW
#   split test -> hanya bila dataset dibuat dengan split.mode=internal (official test, evaluasi akhir)
# Argumen tambahan diteruskan sebagai override, mis.: bash scripts/evaluate_all.sh device=0
set -euo pipefail
cd "$(dirname "$0")/.."
SPLITS="val"
if python -c "import sys,yaml; sys.exit(0 if 'test' in yaml.safe_load(open('datasets/sunrgbd_yolo/sunrgbd_detect.yaml')) else 1)" 2>/dev/null; then
  SPLITS="val test"
fi
for SPLIT in $SPLITS; do
  python -m src.evaluation.evaluate_detection --config configs/evaluation/detection.yaml --include-extra --set split=$SPLIT "$@"
  python -m src.evaluation.evaluate_depth     --config configs/evaluation/depth.yaml     --include-extra --set split=$SPLIT "$@"
done
