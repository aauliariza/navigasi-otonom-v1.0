#!/usr/bin/env bash
# Tahap 8-10: seluruh eksperimen monocular depth estimation.
set -euo pipefail
cd "$(dirname "$0")/.."
SET=("$@")
run() { python -m "$1" --config "$2" ${SET:+--set "${SET[@]}"}; }
run src.training.train_depth    configs/depth/teacher_x.yaml       # 5.2-1 YOLO26x-Depth (teacher)
run src.training.train_depth    configs/depth/baseline.yaml        # 5.2-2 YOLO26n-Depth baseline
run src.training.train_depth_kd configs/depth/output_kd.yaml       # 5.2-3 Depth Output Distillation
run src.training.train_depth_kd configs/depth/feature_kd.yaml      # 5.2-4 Feature Distillation
run src.training.train_depth_kd configs/depth/boundary_kd.yaml     # 5.2-5 Boundary Distillation
run src.training.train_depth_kd configs/depth/relative_kd.yaml     # 5.2-6 Relative Depth Distillation
run src.training.train_depth_kd configs/depth/full_kd.yaml         # 5.2-7 Full Depth KD
run src.training.train_depth    configs/depth/teacher_x_cbam.yaml  # teacher YOLO26x-Depth + CBAM
run src.training.train_depth_kd configs/depth/cbam_kd.yaml         # 5.2-8 CBAM teacher + KD
run src.training.train_depth_kd configs/depth/cbam_full_kd.yaml    # 5.2-9 CBAM teacher + Full Depth KD
