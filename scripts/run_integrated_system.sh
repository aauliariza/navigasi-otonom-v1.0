#!/usr/bin/env bash
# Tahap 14-18: integrasi, evaluasi jarak, evaluasi navigasi, benchmark sistem, interface.
#   bash scripts/run_integrated_system.sh <folder_citra_uji> [contoh_citra_benchmark]
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:?beri folder/berkas citra uji}"
BENCH_IMG="${2:-}"
python -m src.integration.detector_depth_fusion --config configs/system/inference.yaml --source "$SRC"
python -m src.evaluation.evaluate_distance  --config configs/evaluation/distance.yaml
python -m src.evaluation.evaluate_navigation --config configs/evaluation/navigation.yaml
if [[ -n "$BENCH_IMG" ]]; then
  python -m src.evaluation.benchmark --config configs/system/inference.yaml --image "$BENCH_IMG"
fi
echo "Interface: python -m app.main --config configs/system/inference.yaml"
