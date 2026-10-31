# Matriks Eksperimen

Seluruh konfigurasi dapat dijalankan independen (setelah teacher-nya tersedia) dan menulis ke folder ID-nya sendiri
(`outputs/<task>/<ID>/`), sehingga checkpoint tidak pernah saling menimpa.

## 1. Deteksi objek (prompt 5.1; hyperparameter proposal Tabel 3.7)

| ID | Teacher | Metode | Student | Config | Script | Teacher checkpoint |
|---|---|---|---|---|---|---|
| DET-01 | — (YOLO26x dilatih) | Tanpa KD | — | `configs/detection/teacher_x.yaml` | `train_detection` | — |
| DET-02 | — | Baseline tanpa KD | YOLO26n | `baseline.yaml` | `train_detection` | — |
| DET-03 | YOLO26x (DET-01) | Feature KD (score-weighted L2, `dis = 6`, tap 16/19/22) | YOLO26n | `feature_kd.yaml` | `train_detection_kd` | `outputs/detection/DET-01/weights/best.pt` |
| DET-04 | YOLO26x (DET-01) | Prediction KD (soft-label BCE + L1 box) | YOLO26n | `prediction_kd.yaml` | `train_detection_kd` | idem |
| TEACHER-CA | — (YOLO26x + CA dilatih) | Tanpa KD | — | `teacher_x_ca.yaml` | `train_detection` | — |
| DET-05 | YOLO26x + CA | Prediction KD (KD dasar) | YOLO26n | `ca_kd.yaml` | `train_detection_kd` | `outputs/detection/TEACHER-CA/weights/best.pt` |
| DET-06 | YOLO26x + CA | Feature + Prediction KD | YOLO26n | `ca_full_kd.yaml` | `train_detection_kd` | idem |

Hyperparameter bersama (`configs/detection/_common.yaml`): 100 epoch, input 640, AdamW, lr 0,001, lrf 0,01,
weight decay 0,0005, warmup 3 epoch, cosine, early stopping 25, AMP, seed 42, 8 worker; batch 16 (YOLO26n) dan
8 (YOLO26x / YOLO26x + CA). Inferensi: conf 0,25, NMS IoU 0,45 (head one-to-many), max 100 deteksi.

## 2. Monocular depth estimation (prompt 6.1; hyperparameter proposal Tabel 3.8)

| ID | Teacher | output | feature | boundary | relative | Config | Teacher checkpoint |
|---|---|---|---|---|---|---|---|
| DEP-01 | — (YOLO26x-Depth dilatih) | | | | | `configs/depth/teacher_x.yaml` | — |
| DEP-02 | — (baseline YOLO26n-Depth) | | | | | `baseline.yaml` | — |
| DEP-03 | YOLO26x-Depth | 0,5 | | | | `output_kd.yaml` | `outputs/depth/DEP-01/weights/best.pt` |
| DEP-04 | YOLO26x-Depth | | 0,2 | | | `feature_kd.yaml` | idem |
| DEP-05 | YOLO26x-Depth | | | 0,1 | | `boundary_kd.yaml` | idem |
| DEP-06 | YOLO26x-Depth | | | | 0,1 | `relative_kd.yaml` | idem |
| DEP-07 | YOLO26x-Depth | 0,5 | 0,2 | 0,1 | 0,1 | `full_kd.yaml` | idem |
| TEACHER-CBAM | — (YOLO26x-Depth + CBAM dilatih) | | | | | `teacher_x_cbam.yaml` | — |
| DEP-08 | YOLO26x-Depth + CBAM | 0,5 | | | | `cbam_kd.yaml` | `outputs/depth/TEACHER-CBAM/weights/best.pt` |
| DEP-09 | YOLO26x-Depth + CBAM | 0,5 | 0,2 | 0,1 | 0,1 | `cbam_full_kd.yaml` | idem |

Angka = bobot λ komponen (Pers. 3.10 proposal, tanpa suku ROI). Semua student juga memakai loss ground truth
`DepthLoss26` (SILog + gradient matching) dengan bobot 1,0. Batch 16 / 8 dan lr 0,001 (n) / 0,0005 (x). Setelah
training, setiap model dikalibrasi log-affine pada split **val**. Komponen dapat diubah tanpa mengubah
kode, misalnya `--set kd.boundary.enabled=false kd.relative.weight=0.2`.

## 3. Penggunaan split

| Tahap | Split |
|---|---|
| Tahap | Mode `proposal` (dipakai penelitian) | Mode `internal` (opsional) |
|---|---|---|
| Training | `train` = alltrain 5.285 | `train` = alltrain − validasi internal |
| Early stopping, `best.pt`, validasi per epoch | `val` = alltest 5.050 | `val` internal |
| Kalibrasi log-affine | `val` (5.050) | `val` internal |
| Evaluasi untuk EWM-SAW dan pelaporan metrik model | `val` (5.050) | seleksi `val`, laporan `test` |
| Evaluasi jarak / navigasi pada SUN RGB-D | `val` (5.050) | `test` |
| Uji independen jarak / navigasi | data indoor sendiri (`measured`, `scenarios`) | idem |

Deteksi dan depth selalu memakai split yang identik (satu set citra, `meta/splits.json`).

## 4. Struktur output

```
outputs/
├── dataset/                         integrity_report.json, integrity_issues.csv, objects_per_image.*, samples/
├── detection/
│   ├── DET-01 … DET-06, TEACHER-CA/ weights/{best,last}.pt, results.csv (loss GT+KD & metrik val per epoch),
│   │                                results.png, PR/F1/P/R curves, confusion_matrix*.png (Ultralytics), loss_curves.png,
│   │                                args.yaml, config_used.yaml, model_*.yaml (arsitektur), metadata.json
│   │                                (versi, hardware, git commit, split SHA-256, teacher SHA-256), gpu_memory.csv,
│   │                                training_summary.json, evaluation/{val,test}/metrics.json
├── depth/
│   └── DEP-01 … DEP-09, TEACHER-CBAM/  sama + calibration_{best,last}.json
├── evaluation/
│   ├── detection/val/ (+test)       <model>/metrics.json, detection_results.{csv,json}, hardware.json
│   └── depth/val/ (+test)           <model>/{metrics.json, per_image.csv, samples/ (RGB|GT|pred|error), depth_maps/*.npy},
│                                    depth_results.{csv,json}, hardware.json
├── selection/
│   ├── detection_ewm/, depth_ewm/   decision_matrix.csv, ewm_proportions.csv, ewm_weights.csv
│   ├── detection_saw/, depth_saw/   normalized_matrix.csv, saw_scores.csv, ranking.csv, ranking.png, selection_report.{json,md}
│   └── selected_models.json         checkpoint student KD terpilih (input integrasi)
├── integration/
│   ├── distance_evaluation/         distance_per_object.csv, distance_summary.{json,csv}, grafik
│   └── navigation_evaluation/       navigation_per_scenario.csv, confusion_matrix.csv, navigation_summary.{json,csv}
├── inference/
│   ├── images/                      *_result.jpg, objects.csv, decisions.csv, results.json
│   ├── videos/                      *_result.mp4, *_objects.csv, *_decisions.csv, *_summary.json
│   ├── webcam/                      snapshot_*.jpg|json, record_*.mp4, webcam_decisions.csv, webcam_session.json
│   └── interface/                   hasil tombol "Simpan hasil" antarmuka Gradio
└── reports/experiment_summary/      RESEARCH_REPORT.md, experiment_summary_{detection,depth}_{val,test}.csv
```

Experiment ID tetap per konfigurasi; waktu mulai/selesai tercatat di `metadata.json` dan `training_summary.json`.
Metrik yang belum diukur ditulis `NOT MEASURED` pada laporan.

## 5. Protokol pengukuran jarak aktual (`gt_source: measured`)

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

## 6. Protokol skenario navigasi (`gt_source: scenarios`)

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
