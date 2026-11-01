#!/usr/bin/env bash
# Seluruh eksperimen monocular depth estimation DEP-01 ... DEP-14 (teacher sebelum student KD-nya).
# Argumen tambahan diteruskan sebagai override, mis.:  bash scripts/train_all_depth.sh train.device=0
set -euo pipefail
cd "$(dirname "$0")/.."
SET=("$@")
run() { python -m "$1" --config "$2" ${SET:+--set "${SET[@]}"}; }
run src.training.train_depth    configs/depth/teacher_x.yaml        # DEP-01 YOLO26x-Depth (teacher)
run src.training.train_depth    configs/depth/baseline.yaml         # DEP-02 YOLO26n-Depth baseline
run src.training.train_depth_kd configs/depth/output_kd.yaml        # DEP-03 Depth Output Distillation
run src.training.train_depth_kd configs/depth/feature_kd.yaml       # DEP-04 Feature Distillation
run src.training.train_depth_kd configs/depth/boundary_kd.yaml      # DEP-05 Boundary Distillation
run src.training.train_depth_kd configs/depth/relative_kd.yaml      # DEP-06 Relative Depth Distillation
run src.training.train_depth_kd configs/depth/full_kd.yaml          # DEP-07 Output + Feature + Boundary + Relative
run src.training.train_depth    configs/depth/teacher_x_cbam.yaml   # DEP-08 YOLO26x-Depth + CBAM (teacher DEP-10..14)
run src.training.train_depth    configs/depth/student_n_cbam.yaml   # DEP-09 YOLO26n-Depth + CBAM (tanpa KD)
run src.training.train_depth_kd configs/depth/cbam_output_kd.yaml   # DEP-10 CBAM teacher, Depth Output Distillation
run src.training.train_depth_kd configs/depth/cbam_feature_kd.yaml  # DEP-11 CBAM teacher, Feature Distillation
run src.training.train_depth_kd configs/depth/cbam_boundary_kd.yaml # DEP-12 CBAM teacher, Boundary Distillation
run src.training.train_depth_kd configs/depth/cbam_relative_kd.yaml # DEP-13 CBAM teacher, Relative Depth Distillation
run src.training.train_depth_kd configs/depth/cbam_full_kd.yaml     # DEP-14 CBAM teacher, Output + Feature + Boundary + Relative
