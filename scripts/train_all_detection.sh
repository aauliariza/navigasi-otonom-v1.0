#!/usr/bin/env bash
# Seluruh eksperimen deteksi objek DET-01 ... DET-10 (urutan wajib: teacher sebelum student KD-nya).
# Argumen tambahan diteruskan sebagai override, mis.:  bash scripts/train_all_detection.sh train.device=0
set -euo pipefail
cd "$(dirname "$0")/.."
SET=("$@")
run() { python -m "$1" --config "$2" ${SET:+--set "${SET[@]}"}; }
run src.training.train_detection    configs/detection/teacher_x.yaml                # DET-01 YOLO26x (teacher)
run src.training.train_detection    configs/detection/baseline.yaml                 # DET-02 YOLO26n baseline
run src.training.train_detection_kd configs/detection/feature_kd.yaml               # DET-03 YOLO26x -> YOLO26n, Feature KD
run src.training.train_detection_kd configs/detection/prediction_kd.yaml            # DET-04 YOLO26x -> YOLO26n, Prediction KD
run src.training.train_detection_kd configs/detection/feature_prediction_kd.yaml    # DET-05 YOLO26x -> YOLO26n, Feature + Prediction KD
run src.training.train_detection    configs/detection/teacher_x_ca.yaml             # DET-06 YOLO26x + CA (teacher DET-08..10)
run src.training.train_detection    configs/detection/student_n_ca.yaml             # DET-07 YOLO26n + CA (tanpa KD)
run src.training.train_detection_kd configs/detection/ca_feature_kd.yaml            # DET-08 YOLO26x+CA -> YOLO26n, Feature KD
run src.training.train_detection_kd configs/detection/ca_prediction_kd.yaml         # DET-09 YOLO26x+CA -> YOLO26n, Prediction KD
run src.training.train_detection_kd configs/detection/ca_feature_prediction_kd.yaml # DET-10 YOLO26x+CA -> YOLO26n, Feature + Prediction KD
