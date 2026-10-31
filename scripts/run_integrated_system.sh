#!/usr/bin/env bash
# Integrasi, evaluasi jarak, evaluasi navigasi, benchmark sistem.
#   bash scripts/run_integrated_system.sh <folder_citra_uji> [contoh_citra_benchmark]
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="${1:?beri folder/berkas citra uji}"
BENCH_IMG="${2:-}"
python -m src.inference.predict_image --source "$SRC" --output outputs/inference/images
python -m src.evaluation.evaluate_distance  --config configs/evaluation/distance.yaml
python -m src.evaluation.evaluate_navigation --config configs/evaluation/navigation.yaml
if [[ -n "$BENCH_IMG" ]]; then
  python -m src.evaluation.benchmark --config configs/system/inference.yaml --image "$BENCH_IMG" --set output_dir=outputs/integration
fi
python -m src.utils.report --root outputs --out outputs/reports/experiment_summary/RESEARCH_REPORT.md
echo "Interface: python -m app.main --config configs/system/inference.yaml"
