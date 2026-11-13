#!/usr/bin/env bash
# Skenario SCRATCH: seluruh eksperimen detection tanpa bobot pretrained (teacher sebelum student KD-nya).
# Run yang selesai dilewati; run yang terhenti dilanjutkan dari last.pt. Argumen tambahan = override,
# mis.: bash scenarios/scratch/train_all_detection.sh train.device=0
source "$(dirname "$0")/common.sh"
refuse_if_training_running
SET=("$@")
run_experiment src.training.train_detection detection/teacher_x.yaml DET-01  # YOLO26x (teacher)
run_experiment src.training.train_detection detection/baseline.yaml DET-02  # YOLO26n baseline
run_experiment src.training.train_detection_kd detection/feature_kd.yaml DET-03  # YOLO26x -> YOLO26n, Feature KD
run_experiment src.training.train_detection_kd detection/prediction_kd.yaml DET-04  # YOLO26x -> YOLO26n, Prediction KD
run_experiment src.training.train_detection_kd detection/feature_prediction_kd.yaml DET-05  # YOLO26x -> YOLO26n, Feature + Prediction KD
run_experiment src.training.train_detection detection/teacher_x_ca.yaml DET-06  # YOLO26x + CA (teacher DET-08..10)
run_experiment src.training.train_detection detection/student_n_ca.yaml DET-07  # YOLO26n + CA (tanpa KD)
run_experiment src.training.train_detection_kd detection/ca_feature_kd.yaml DET-08  # YOLO26x+CA -> YOLO26n, Feature KD
run_experiment src.training.train_detection_kd detection/ca_prediction_kd.yaml DET-09  # YOLO26x+CA -> YOLO26n, Prediction KD
run_experiment src.training.train_detection_kd detection/ca_feature_prediction_kd.yaml DET-10  # YOLO26x+CA -> YOLO26n, Feature + Prediction KD
