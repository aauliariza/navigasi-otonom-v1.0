#!/usr/bin/env bash
# Smoke test end-to-end SELURUH pipeline dengan data SINTETIS berformat SUN RGB-D (CPU, beberapa menit).
# Menjalankan 10 eksperimen deteksi dan 14 eksperimen depth (1 epoch, imgsz 96, arsitektur teacher
# diganti varian n agar cepat), evaluasi, EWM-SAW, integrasi, evaluasi jarak & navigasi.
# Angka yang dihasilkan TIDAK bermakna ilmiah; tujuannya hanya memastikan semua tahap dapat berjalan.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/smoke_e2e
rm -rf "$OUT"
python scripts/data/make_synthetic_sunrgbd.py --out $OUT/raw --train 10 --test 4
python -m src.data.prepare_sunrgbd --set source.sunrgbd_root=$OUT/raw/SUNRGBD \
  source.split_file=$OUT/raw/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat source.bbox_meta=$OUT/raw/SUNRGBDMeta2DBB_v2.mat \
  output.root=$OUT/data output.detect_yaml=$OUT/data/detect.yaml output.depth_yaml=$OUT/data/depth.yaml
python scripts/data/validate_dataset.py --set output.root=$OUT/data output.report_dir=$OUT/dataset \
  split.expected_counts.train_official=10 split.expected_counts.alltest=4
C="data.integrity_report=$OUT/dataset/integrity_report.json model.weights=null train.epochs=1 train.batch=4 train.imgsz=96 train.workers=0 train.device=cpu train.amp=false train.plots=false output.root=$OUT/runs"
D="$C data.yaml=$OUT/data/detect.yaml"
P="$C data.yaml=$OUT/data/depth.yaml"
# Deteksi DET-01 ... DET-10 (teacher x diganti arsitektur n agar cepat; teacher diberikan eksplisit)
python -m src.training.train_detection --config configs/detection/teacher_x.yaml    --set $D model.arch=yolo26n.yaml   # DET-01
python -m src.training.train_detection --config configs/detection/baseline.yaml     --set $D                           # DET-02
python -m src.training.train_detection --config configs/detection/teacher_x_ca.yaml --set $D model.arch=yolo26n.yaml   # DET-06
python -m src.training.train_detection --config configs/detection/student_n_ca.yaml --set $D                           # DET-07
T=$OUT/runs/detection/DET-01/weights/best.pt
TC=$OUT/runs/detection/DET-06/weights/best.pt
for c in feature_kd prediction_kd feature_prediction_kd; do                                                            # DET-03..05
  python -m src.training.train_detection_kd --config configs/detection/$c.yaml --set $D kd.teacher_weights=$T
done
for c in ca_feature_kd ca_prediction_kd ca_feature_prediction_kd; do                                                   # DET-08..10
  python -m src.training.train_detection_kd --config configs/detection/$c.yaml --set $D kd.teacher_weights=$TC
done
# Depth DEP-01 ... DEP-14
python -m src.training.train_depth --config configs/depth/teacher_x.yaml      --set $P model.arch=yolo26n-depth.yaml    # DEP-01
python -m src.training.train_depth --config configs/depth/baseline.yaml       --set $P                                  # DEP-02
python -m src.training.train_depth --config configs/depth/teacher_x_cbam.yaml --set $P model.arch=yolo26n-depth.yaml    # DEP-08
python -m src.training.train_depth --config configs/depth/student_n_cbam.yaml --set $P                                  # DEP-09
T=$OUT/runs/depth/DEP-01/weights/best.pt
TC=$OUT/runs/depth/DEP-08/weights/best.pt
for c in output_kd feature_kd boundary_kd relative_kd full_kd; do                                                     # DEP-03..07
  python -m src.training.train_depth_kd --config configs/depth/$c.yaml --set $P kd.teacher_weights=$T
done
for c in cbam_output_kd cbam_feature_kd cbam_boundary_kd cbam_relative_kd cbam_full_kd; do                            # DEP-10..14
  python -m src.training.train_depth_kd --config configs/depth/$c.yaml --set $P kd.teacher_weights=$TC
done
E="device=cpu imgsz=96 batch=4 benchmark.warmup=2 benchmark.iters=5 experiment_overrides=[output.root=$OUT/runs]"
# seleksi pada validasi internal, evaluasi akhir pada official test
for SPLIT in val; do  # mode split proposal: tidak ada split test
  python -m src.evaluation.evaluate_detection --include-extra --set split=$SPLIT data_yaml=$OUT/data/detect.yaml "output_dir=$OUT/eval/detection/{split}" $E
  python -m src.evaluation.evaluate_depth     --include-extra --set split=$SPLIT data_yaml=$OUT/data/depth.yaml  "output_dir=$OUT/eval/depth/{split}" $E
done
python -m src.optimization.model_selection --set detection.results=$OUT/eval/detection/val/detection_results.csv \
  detection.output_dir=$OUT/selection depth.results=$OUT/eval/depth/val/depth_results.csv \
  depth.output_dir=$OUT/selection selected_models_path=$OUT/selection/selected_models.json
SYS="models.selected_models=$OUT/selection/selected_models.json output_dir=$OUT/system detection.imgsz=96 depth.imgsz=96 detection.conf=0.001"
python -m src.inference.predict_image --source $OUT/data/images/val --output $OUT/inference/images --set $SYS
python - "$OUT" <<'PY'
import sys, glob, cv2
out = sys.argv[1]
frames = [cv2.imread(p) for p in sorted(glob.glob(f"{out}/data/images/val/*.jpg"))]
h, w = frames[0].shape[:2]
vw = cv2.VideoWriter(f"{out}/clip.avi", cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (w, h))
for f in frames * 3:
    vw.write(cv2.resize(f, (w, h)))
vw.release()
PY
python -m src.inference.predict_video --source $OUT/clip.avi --output $OUT/inference/videos --set $SYS
SO="system_overrides=[${SYS// /,}]"
python -m src.evaluation.evaluate_distance  --set "$SO" data_yaml=$OUT/data/depth.yaml output_dir=$OUT/eval/distance
python -m src.evaluation.evaluate_navigation --set "$SO" gt_source=sunrgbd data_yaml=$OUT/data/depth.yaml output_dir=$OUT/eval/navigation
python -m src.evaluation.benchmark --image "$(ls $OUT/data/images/val/*.jpg | head -1)" --warmup 2 --iters 5 --set $SYS
python -m src.utils.report --root $OUT --out $OUT/reports/experiment_summary/RESEARCH_REPORT.md
echo "SMOKE TEST OK -> $OUT"
