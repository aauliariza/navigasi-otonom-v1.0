# Daftar Eksperimen dan Protokol Pengujian

## 1. Object detection (proposal Tabel 3.7; `configs/detection/`)

| No | Experiment ID | Config | Teacher | KD | Script |
|---|---|---|---|---|---|
| T | `det_00_yolo26x_ca_teacher` | `teacher_x_ca.yaml` | — (YOLO26x + CA, dilatih) | — | `train_detection` |
| 5.1-1 | `det_01_yolo26x_teacher` | `teacher_x.yaml` | — | — | `train_detection` |
| 5.1-2 | `det_02_yolo26n_baseline` | `baseline.yaml` | — | — | `train_detection` |
| 5.1-3 | `det_03_yolo26n_feature_kd` | `feature_kd.yaml` | YOLO26x | Feature (`dis = 6`) | `train_detection_kd` |
| 5.1-4 | `det_04_yolo26n_prediction_kd` | `prediction_kd.yaml` | YOLO26x | Prediction | `train_detection_kd` |
| 5.1-5 | `det_05_yolo26n_ca_teacher_kd` | `ca_kd.yaml` | YOLO26x + CA | Feature (`dis = 6`) | `train_detection_kd` |
| 5.1-6 | `det_06_yolo26n_ca_feature_prediction_kd` | `ca_full_kd.yaml` | YOLO26x + CA | Feature + Prediction | `train_detection_kd` |

Hyperparameter bersama (`_common.yaml`):

- 100 epoch, input 640.
- AdamW, lr 0,001, lrf 0,01, weight decay 0,0005.
- Warmup 3 epoch, scheduler cosine.
- Early stopping 25, AMP, seed 42, 8 worker.
- Batch 16 (n) / 8 (x).

## 2. Monocular depth (proposal Tabel 3.8; `configs/depth/`)

| No | Experiment ID | Config | Teacher | Komponen KD |
|---|---|---|---|---|
| T | `dep_00_yolo26x_depth_cbam_teacher` | `teacher_x_cbam.yaml` | — (x-Depth + CBAM) | — |
| 5.2-1 | `dep_01_yolo26x_depth_teacher` | `teacher_x.yaml` | — | — |
| 5.2-2 | `dep_02_yolo26n_depth_baseline` | `baseline.yaml` | — | — |
| 5.2-3 | `dep_03_yolo26n_depth_output_kd` | `output_kd.yaml` | x-Depth | output (0,5) |
| 5.2-4 | `dep_04_yolo26n_depth_feature_kd` | `feature_kd.yaml` | x-Depth | feature (0,2) |
| 5.2-5 | `dep_05_yolo26n_depth_boundary_kd` | `boundary_kd.yaml` | x-Depth | boundary (0,1) |
| 5.2-6 | `dep_06_yolo26n_depth_relative_kd` | `relative_kd.yaml` | x-Depth | relative (0,1) |
| 5.2-7 | `dep_07_yolo26n_depth_full_kd` | `full_kd.yaml` | x-Depth | keempatnya |
| 5.2-8 | `dep_08_yolo26n_depth_cbam_teacher_kd` | `cbam_kd.yaml` | x-Depth + CBAM | output (0,5) |
| 5.2-9 | `dep_09_yolo26n_depth_cbam_full_kd` | `cbam_full_kd.yaml` | x-Depth + CBAM | keempatnya |

Batch 16 (n) / 8 (x) dan lr 0,001 (n) / 0,0005 (x). Setelah training, semua model depth dikalibrasi log-affine
pada split train.

Komponen KD dapat diaktifkan atau dinonaktifkan tanpa mengubah kode, contohnya:
`--set kd.boundary.enabled=false kd.relative.weight=0.2`.

## 3. Struktur output

```
outputs/
├── dataset/                         integrity_report.json, issues, sampel visual
├── detection/<experiment_id>/       weights/{best,last}.pt, results.csv (loss GT+KD & metrik val per epoch),
│                                    results.png, loss_curves.png, args.yaml, config_used.yaml, metadata.json
│                                    (git commit, versi, hardware, SHA-256 teacher), gpu_memory.csv,
│                                    training_summary.json, model_*.yaml
├── depth/<experiment_id>/           sama + calibration_{best,last}.json
├── evaluation/detection|depth/      <model>/metrics.json, *_results.csv/json, hardware.json, sampel depth
├── model_selection/detection|depth/ decision_matrix, ewm_weights, normalized_matrix, saw_ranking, selection_report.{json,md}
├── model_selection/selected_models.json
├── integrated_system/               *_result.jpg, objects.csv, decisions.csv, results.json, benchmark_system.json
├── evaluation/distance/             distance_per_object.csv, distance_summary.{json,csv}
├── evaluation/navigation/           navigation_per_scenario.csv, confusion_matrix.csv, navigation_summary.{json,csv}
└── RESEARCH_REPORT.md
```

Experiment ID tetap per konfigurasi, sedangkan waktu mulai dan selesai tercatat di `metadata.json` dan
`training_summary.json`. Setiap checkpoint dapat ditelusuri ke `config_used.yaml` dan `args.yaml` di folder yang
sama.

## 4. Protokol pengukuran jarak aktual (`gt_source: measured`)

1. Susun skenario indoor dengan objek statis dan pencahayaan stabil.
2. Ambil foto RGB dengan webcam yang sama dengan pengujian (Logitech C270).
3. Ukur jarak sepanjang sumbu optik kamera ke permukaan objek dengan meteran atau laser distance meter. Ini adalah
   **jarak aksial**, sesuai mode utama sistem.
4. Gambar box di sekitar objek yang diukur pada citra.
5. Isi `data_collection/distance_measurements.csv` (template:
   `data_collection/distance_measurements_template.csv`):
   `image,x1,y1,x2,y2,gt_distance_m,object_name,notes`. Path `image` relatif terhadap folder CSV.
6. Jalankan `python -m src.evaluation.evaluate_distance --set gt_source=measured`.

Box yang tidak tercocokkan dengan deteksi (IoU < 0,5) dilaporkan sebagai `not_detected` dan tidak ikut dalam
MAE, RMSE, maupun MAPE.

## 5. Protokol skenario navigasi (`gt_source: scenarios`)

1. Untuk setiap skenario, tentukan arah sebenarnya berdasarkan posisi dan jarak objek terukur, dengan aturan
   proposal:
   - ROI 60% tengah dibagi lima sektor.
   - Sektor diblokir oleh objek berjarak < 1,0 m.
   - Prioritas CTR → L → R → FL → FR.
2. Isi `data_collection/navigation_scenarios.csv` (template tersedia):
   `scenario_id,image,gt_direction,description`, dengan `gt_direction` ∈ {FORWARD, TURN_LEFT, TURN_RIGHT, STOP}.
3. Jalankan `python -m src.evaluation.evaluate_navigation`.

Mode `gt_source: sunrgbd` menurunkan arah referensi dari box dan depth ground truth SUN RGB-D. Mode ini berguna
sebagai evaluasi tambahan berskala besar, tetapi tidak menggantikan skenario nyata.

**Jangan mengisi template dengan nilai karangan.** Semua nilai harus berasal dari pengukuran nyata.
