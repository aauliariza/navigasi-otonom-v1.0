#!/usr/bin/env bash
# Smoke test end-to-end SELURUH pipeline dengan data SINTETIS berformat SUN RGB-D (CPU, beberapa menit).
# Menjalankan 6 + 1 eksperimen deteksi dan 9 + 1 eksperimen depth (1 epoch, imgsz 96, arsitektur teacher
# diganti varian n agar cepat), evaluasi, EWM-SAW, integrasi, evaluasi jarak & navigasi.
# Angka yang dihasilkan TIDAK bermakna ilmiah; tujuannya hanya memastikan semua tahap dapat berjalan.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/smoke_e2e
rm -rf "$OUT"
python scripts/data/make_synthetic_sunrgbd.py --out $OUT/raw --train 8 --val 4
python -m src.data.prepare_sunrgbd --set source.sunrgbd_root=$OUT/raw/SUNRGBD \
  source.split_file=$OUT/raw/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat source.bbox_meta=$OUT/raw/SUNRGBDMeta2DBB_v2.mat \
  output.root=$OUT/data output.detect_yaml=$OUT/data/detect.yaml output.depth_yaml=$OUT/data/depth.yaml
python scripts/data/validate_dataset.py --set output.root=$OUT/data output.report_dir=$OUT/dataset \
  split.expected_counts.train=8 split.expected_counts.val=4
C="data.integrity_report=$OUT/dataset/integrity_report.json model.weights=null train.epochs=1 train.batch=4 train.imgsz=96 train.workers=0 train.device=cpu train.amp=false train.plots=false output.root=$OUT/runs"
D="$C data.yaml=$OUT/data/detect.yaml"
P="$C data.yaml=$OUT/data/depth.yaml"
python -m src.training.train_detection --config configs/detection/teacher_x.yaml    --set $D model.arch=yolo26n.yaml
python -m src.training.train_detection --config configs/detection/teacher_x_ca.yaml --set $D model.arch=yolo26n.yaml
python -m src.training.train_detection --config configs/detection/baseline.yaml     --set $D
T=$OUT/runs/detection/det_01_yolo26x_teacher/weights/best.pt
TC=$OUT/runs/detection/det_00_yolo26x_ca_teacher/weights/best.pt
python -m src.training.train_detection_kd --config configs/detection/feature_kd.yaml    --set $D kd.teacher_weights=$T
python -m src.training.train_detection_kd --config configs/detection/prediction_kd.yaml --set $D kd.teacher_weights=$T
python -m src.training.train_detection_kd --config configs/detection/ca_kd.yaml         --set $D kd.teacher_weights=$TC
python -m src.training.train_detection_kd --config configs/detection/ca_full_kd.yaml    --set $D kd.teacher_weights=$TC
python -m src.training.train_depth --config configs/depth/teacher_x.yaml      --set $P model.arch=yolo26n-depth.yaml
python -m src.training.train_depth --config configs/depth/teacher_x_cbam.yaml --set $P model.arch=yolo26n-depth.yaml
python -m src.training.train_depth --config configs/depth/baseline.yaml       --set $P
T=$OUT/runs/depth/dep_01_yolo26x_depth_teacher/weights/best.pt
TC=$OUT/runs/depth/dep_00_yolo26x_depth_cbam_teacher/weights/best.pt
for c in output_kd feature_kd boundary_kd relative_kd full_kd; do
  python -m src.training.train_depth_kd --config configs/depth/$c.yaml --set $P kd.teacher_weights=$T
done
for c in cbam_kd cbam_full_kd; do
  python -m src.training.train_depth_kd --config configs/depth/$c.yaml --set $P kd.teacher_weights=$TC
done
E="device=cpu imgsz=96 batch=4 benchmark.warmup=2 benchmark.iters=5 experiment_overrides=[output.root=$OUT/runs]"
python -m src.evaluation.evaluate_detection --include-extra --set data_yaml=$OUT/data/detect.yaml output_dir=$OUT/eval/detection $E
python -m src.evaluation.evaluate_depth     --include-extra --set data_yaml=$OUT/data/depth.yaml  output_dir=$OUT/eval/depth $E
python -m src.optimization.model_selection --set detection.results=$OUT/eval/detection/detection_results.csv \
  detection.output_dir=$OUT/selection/detection depth.results=$OUT/eval/depth/depth_results.csv \
  depth.output_dir=$OUT/selection/depth selected_models_path=$OUT/selection/selected_models.json
SYS="models.selected_models=$OUT/selection/selected_models.json output_dir=$OUT/system detection.imgsz=96 depth.imgsz=96 detection.conf=0.001"
python -m src.integration.detector_depth_fusion --source $OUT/data/images/val --set $SYS
SO="system_overrides=[${SYS// /,}]"
python -m src.evaluation.evaluate_distance  --set "$SO" data_yaml=$OUT/data/depth.yaml output_dir=$OUT/eval/distance
python -m src.evaluation.evaluate_navigation --set "$SO" gt_source=sunrgbd data_yaml=$OUT/data/depth.yaml output_dir=$OUT/eval/navigation
python -m src.evaluation.benchmark --image "$(ls $OUT/data/images/val/*.jpg | head -1)" --warmup 2 --iters 5 --set $SYS
python -m src.utils.report --root $OUT --out $OUT/RESEARCH_REPORT.md
echo "SMOKE TEST OK -> $OUT"
