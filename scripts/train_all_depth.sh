#!/usr/bin/env bash
# Tahap 8-10: seluruh eksperimen monocular depth estimation.
set -euo pipefail
cd "$(dirname "$0")/.."
SET=("$@")
run() { python -m "$1" --config "$2" ${SET:+--set "${SET[@]}"}; }
run src.training.train_depth    configs/depth/teacher_x.yaml       # DEP-01 YOLO26x-Depth (teacher)
run src.training.train_depth    configs/depth/baseline.yaml        # DEP-02 YOLO26n-Depth baseline
run src.training.train_depth_kd configs/depth/output_kd.yaml       # DEP-03 Depth Output Distillation
run src.training.train_depth_kd configs/depth/feature_kd.yaml      # DEP-04 Feature Distillation
run src.training.train_depth_kd configs/depth/boundary_kd.yaml     # DEP-05 Boundary Distillation
run src.training.train_depth_kd configs/depth/relative_kd.yaml     # DEP-06 Relative Depth Distillation
run src.training.train_depth_kd configs/depth/full_kd.yaml         # DEP-07 Full Depth KD
run src.training.train_depth    configs/depth/teacher_x_cbam.yaml  # TEACHER-CBAM: YOLO26x-Depth + CBAM
run src.training.train_depth_kd configs/depth/cbam_kd.yaml         # DEP-08 CBAM teacher + Depth Output KD
run src.training.train_depth_kd configs/depth/cbam_full_kd.yaml    # DEP-09 CBAM teacher + Full Depth KD
