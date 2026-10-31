#!/usr/bin/env bash
# Tahap 4-6: seluruh eksperimen deteksi objek (urutan wajib: teacher dulu, lalu student).
# Argumen tambahan diteruskan sebagai override, mis.:  bash scripts/train_all_detection.sh train.device=0
set -euo pipefail
cd "$(dirname "$0")/.."
SET=("$@")
run() { python -m "$1" --config "$2" ${SET:+--set "${SET[@]}"}; }
run src.training.train_detection    configs/detection/teacher_x.yaml      # DET-01 YOLO26x (teacher)
run src.training.train_detection    configs/detection/baseline.yaml       # DET-02 YOLO26n baseline
run src.training.train_detection_kd configs/detection/feature_kd.yaml     # DET-03 Feature KD
run src.training.train_detection_kd configs/detection/prediction_kd.yaml  # DET-04 Prediction KD
run src.training.train_detection    configs/detection/teacher_x_ca.yaml   # TEACHER-CA: YOLO26x + Coordinate Attention
run src.training.train_detection_kd configs/detection/ca_kd.yaml          # DET-05 CA teacher + Prediction KD
run src.training.train_detection_kd configs/detection/ca_full_kd.yaml     # DET-06 CA teacher + Feature + Prediction KD
