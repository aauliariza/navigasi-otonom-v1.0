#!/usr/bin/env bash
# Skenario SCRATCH: seluruh eksperimen depth tanpa bobot pretrained (teacher sebelum student KD-nya).
# Run yang selesai dilewati; run yang terhenti dilanjutkan dari last.pt. Argumen tambahan = override,
# mis.: bash scenarios/scratch/train_all_depth.sh train.device=0
source "$(dirname "$0")/common.sh"
refuse_if_training_running
SET=("$@")
run_experiment src.training.train_depth depth/teacher_x.yaml DEP-01  # YOLO26x-Depth (teacher)
run_experiment src.training.train_depth depth/baseline.yaml DEP-02  # YOLO26n-Depth baseline
run_experiment src.training.train_depth_kd depth/output_kd.yaml DEP-03  # Depth Output Distillation
run_experiment src.training.train_depth_kd depth/feature_kd.yaml DEP-04  # Feature Distillation
run_experiment src.training.train_depth_kd depth/boundary_kd.yaml DEP-05  # Boundary Distillation
run_experiment src.training.train_depth_kd depth/relative_kd.yaml DEP-06  # Relative Depth Distillation
run_experiment src.training.train_depth_kd depth/full_kd.yaml DEP-07  # Output + Feature + Boundary + Relative
run_experiment src.training.train_depth depth/teacher_x_cbam.yaml DEP-08  # YOLO26x-Depth + CBAM (teacher DEP-10..14)
run_experiment src.training.train_depth depth/student_n_cbam.yaml DEP-09  # YOLO26n-Depth + CBAM (tanpa KD)
run_experiment src.training.train_depth_kd depth/cbam_output_kd.yaml DEP-10  # CBAM teacher, Depth Output Distillation
run_experiment src.training.train_depth_kd depth/cbam_feature_kd.yaml DEP-11  # CBAM teacher, Feature Distillation
run_experiment src.training.train_depth_kd depth/cbam_boundary_kd.yaml DEP-12  # CBAM teacher, Boundary Distillation
run_experiment src.training.train_depth_kd depth/cbam_relative_kd.yaml DEP-13  # CBAM teacher, Relative Depth Distillation
run_experiment src.training.train_depth_kd depth/cbam_full_kd.yaml DEP-14  # CBAM teacher, Output + Feature + Boundary + Relative
