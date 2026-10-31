#!/usr/bin/env bash
# Tahap 4-6: seluruh eksperimen deteksi objek (urutan wajib: teacher dulu, lalu student).
# Argumen tambahan diteruskan sebagai override, mis.:  bash scripts/train_all_detection.sh train.device=0
set -euo pipefail
cd "$(dirname "$0")/.."
SET=("$@")
run() { python -m "$1" --config "$2" ${SET:+--set "${SET[@]}"}; }
run src.training.train_detection    configs/detection/teacher_x.yaml      # 5.1-1 YOLO26x (teacher)
run src.training.train_detection    configs/detection/baseline.yaml       # 5.1-2 YOLO26n baseline
run src.training.train_detection_kd configs/detection/feature_kd.yaml     # 5.1-3 Feature KD
run src.training.train_detection_kd configs/detection/prediction_kd.yaml  # 5.1-4 Prediction KD
run src.training.train_detection    configs/detection/teacher_x_ca.yaml   # teacher YOLO26x + CA
run src.training.train_detection_kd configs/detection/ca_kd.yaml          # 5.1-5 CA teacher + KD
run src.training.train_detection_kd configs/detection/ca_full_kd.yaml     # 5.1-6 CA teacher + Feature + Prediction KD
