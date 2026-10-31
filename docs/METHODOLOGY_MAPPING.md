# Pemetaan Metodologi ke Kode

Dokumen ini menunjukkan di mana setiap bagian proposal (BAB 3) dan setiap bagian prompt diimplementasikan.
Keputusan yang menyimpang dari proposal ada di [EXPERIMENT_DESIGN_DECISIONS.md](EXPERIMENT_DESIGN_DECISIONS.md).

## 1. Sub-bab proposal → modul

| Proposal | Isi | Modul / CLI | Config |
|---|---|---|---|
| 3.2.3.1, Tabel 3.6 | Dataset SUN RGB-D, pembagian resmi | `src/data/prepare_sunrgbd.py`, `src/data/create_split.py` | `configs/dataset/sunrgbd.yaml` |
| 3.2.3.2, Pers. 3.1–3.2 | Pairing RGB-depth, dekode uint16, meter | `src/data/pair_rgb_depth.py`, `src/data/prepare_depth.py` | idem |
| 3.2.3.3, Pers. 3.3–3.8 | Dataset deteksi satu kelas, YOLO, sisi ≥ 4 px | `src/data/prepare_annotations.py` | idem |
| Tabel 3.7 | Hyperparameter deteksi | `configs/detection/_common.yaml` (+ batch 8 pada teacher) | — |
| Tabel 3.8 | Hyperparameter depth | `configs/depth/_common.yaml` (+ batch 8, lr 0,0005 pada teacher) | — |
| 3.2.4.1 | YOLO26x / YOLO26n, conf 0,25, NMS 0,45, max 100 | `src/training/train_detection.py` | `configs/detection/{teacher_x,baseline}.yaml`, `configs/system/inference.yaml` |
| 3.2.4.2, Pers. 2.5–2.7 | KD deteksi (framework, `dis = 6`) | `models/detection/detection_teacher_student.py`, `src/losses/detection_kd.py`, `src/training/train_detection_kd.py` | `configs/detection/*kd*.yaml` |
| 3.2.4.3, Pers. 2.4 | YOLO26-Depth, rentang 0,001–10 m, log-affine | `src/training/train_depth.py`, `models/depth/yolo26_depth.py` | `configs/depth/{teacher_x,baseline}.yaml` |
| 3.2.4.4, Pers. 2.8–2.14, 3.9–3.10 | KD depth (4 komponen, tanpa ROI) | `models/depth/depth_teacher_student.py`, `src/losses/*`, `src/training/train_depth_kd.py` | `configs/depth/*kd*.yaml` |
| 3.2.4.5, Pers. 3.11–3.12 | Integrasi, penyesuaian koordinat | `src/integration/detector_depth_fusion.py`, `spatial_alignment.py` | `configs/system/inference.yaml` |
| 3.2.4.6, Pers. 3.13–3.19 | Inner ROI, filter, persentil, median, R_valid | `src/integration/inner_roi.py`, `object_distance.py` | idem (`distance:`) |
| 3.2.4.7, Pers. 3.20 | ROI navigasi 60%, 5 sektor, blocking, prioritas | `src/navigation/*` | idem (`navigation:`) |
| 3.2.5.1, Pers. 3.21–3.33 | Evaluasi deteksi, efisiensi, EWM-SAW | `src/evaluation/evaluate_detection.py`, `benchmark.py`, `src/optimization/*` | `configs/evaluation/{detection,selection}.yaml` |
| 3.2.5.2, Pers. 3.34–3.38 | Evaluasi depth, EWM-SAW | `src/evaluation/evaluate_depth.py`, `metrics.py` | `configs/evaluation/{depth,selection}.yaml` |
| 3.2.5.3, Pers. 3.39–3.43 | Akurasi jarak MAE/RMSE/MAPE | `src/evaluation/evaluate_distance.py` | `configs/evaluation/distance.yaml` |
| 3.2.5.4, Pers. 3.44–3.45 | Akurasi navigasi | `src/evaluation/evaluate_navigation.py` | `configs/evaluation/navigation.yaml` |

## 2. Pemetaan persamaan proposal ke kode

| Persamaan | Isi | Implementasi |
|---|---|---|
| 2.4 | log-affine `exp(a log D + b)` | `models/depth/yolo26_depth.py::fit_log_affine`, buffer `cal_a`/`cal_b` |
| 2.8, 3.9–3.10 | total loss KD depth | `src/losses/combined_loss.py` (tanpa ROI) |
| 2.9 | Depth output distillation | `src/losses/depth_output_loss.py` |
| 2.10 | Feature distillation | `src/losses/feature_loss.py::feature_distillation` |
| 2.11–2.12 | Boundary distillation | `src/losses/boundary_loss.py` |
| 2.13–2.14 | Relative depth distillation | `src/losses/relative_depth_loss.py` |
| 2.7 | `L_OD = L_det + λ L_KD` | `models/detection/detection_teacher_student.py::loss` |
| 3.1–3.2 | dekode depth, mm→m | `src/data/prepare_depth.py` |
| 3.3–3.8 | konversi box ke YOLO | `src/data/prepare_annotations.py::xyxy_to_yolo` |
| 3.11–3.12 | spatial alignment | `src/integration/spatial_alignment.py` |
| 3.13–3.17 | inner ROI 60% | `src/integration/inner_roi.py` |
| 3.18–3.19 | median, R_valid | `src/integration/object_distance.py` |
| 3.20, 3.44 | sektor 0,12 W, overlap ratio | `src/navigation/sector_division.py`, `obstacle_analysis.py` |
| 3.21–3.24 | mAP, P, R, F1 | `src/evaluation/evaluate_detection.py`, `metrics.py` |
| 3.25–3.26 | FLOPs, FPS | `src/evaluation/benchmark.py` |
| 3.27–3.33 | SAW + EWM | `src/optimization/*` |
| 3.34–3.38 | metrik depth | `src/evaluation/metrics.py::DepthMetricAccumulator` |
| 3.39–3.43 | error jarak, MAE, RMSE, MAPE | `src/evaluation/metrics.py::distance_metrics`, `evaluate_distance.py` |
| 3.45 | akurasi navigasi | `src/evaluation/metrics.py::navigation_accuracy`, `evaluate_navigation.py` |


## 3. Bagian prompt → implementasi

| Prompt | Implementasi |
|---|---|
| 5 (DET-01…06) | `configs/detection/*.yaml`, `scripts/train_all_detection.sh`, [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) |
| 6 (DEP-01…09) | `configs/depth/*.yaml`, `scripts/train_all_depth.sh` |
| 7.1 pairing | `src/data/pair_rgb_depth.py` (identitas scene, bukan urutan berkas) |
| 7.2 split + validasi internal | `src/data/create_split.py`, `meta/splits.json`, cek kebocoran di `src/data/validate_dataset.py` |
| 7.3 depth | `src/data/prepare_depth.py` |
| 7.4 anotasi | `src/data/prepare_annotations.py` |
| 7.5 augmentasi | augmentasi Ultralytics (deteksi: mosaic, HSV, flip, scale, translate; depth: transformasi geometris sinkron RGB-depth oleh `DepthDataset`); semua parameter dapat diatur di `train:` (`hsv_*`, `degrees`, `translate`, `scale`, `fliplr`, `flipud`, `mosaic`, `mixup`, `close_mosaic`). Validasi/test tanpa augmentasi |
| 8 training, resume, AMP, seed, early stopping, log GPU | `src/training/trainer_utils.py`, opsi `--resume` |
| 9 evaluasi deteksi | `src/evaluation/evaluate_detection.py` |
| 10 evaluasi depth + kalibrasi pada val | `src/evaluation/evaluate_depth.py`, `models/depth/yolo26_depth.py` |
| 11 EWM + SAW | `src/optimization/*` → `outputs/selection/{detection,depth}_{ewm,saw}/` |
| 12 integrasi dan jarak | `src/integration/*`, `src/evaluation/evaluate_distance.py` |
| 13 navigasi | `src/navigation/*`, `src/evaluation/evaluate_navigation.py` |
| 14 gambar / video / webcam / antarmuka | `src/inference/{predict_image,predict_video,webcam}.py`, `app/main.py` |
| 15 laporan per ID, `NOT MEASURED` | `src/utils/report.py` |
| 19 QA | `tests/`, `scripts/smoke_test.sh` |
| 20 reproducibility | `metadata.json` (config, split SHA-256, seed, versi software, hardware, git commit), `config_used.yaml`, `training_summary.json` |
