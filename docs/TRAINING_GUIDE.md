# Panduan Training

Semua perintah dijalankan dari root repository setelah dataset lulus validasi
(`outputs/dataset/integrity_report.json` berisi `"passed": true`; training menolak berjalan bila tidak).
GPU dipilih dengan `--set train.device=0`. Setiap nilai YAML dapat ditimpa dengan `--set kunci.sub=nilai`.

## 1. Urutan dan ketergantungan

Teacher harus selesai lebih dulu: script KD membaca `outputs/<task>/<teacher-ID>/weights/best.pt`, menolak
berjalan bila berkas itu tidak ada, dan mencatat SHA-256-nya di `metadata.json` student.

```
DET-01 ──► DET-03, DET-04, DET-05            DET-06 (YOLO26x+CA) ──► DET-08, DET-09, DET-10
DET-02, DET-07 (YOLO26n+CA): mandiri
DEP-01 ──► DEP-03 … DEP-07                   DEP-08 (YOLO26x-Depth+CBAM) ──► DEP-10 … DEP-14
DEP-02, DEP-09 (YOLO26n-Depth+CBAM): mandiri
```

## 2. Deteksi objek

| Langkah | ID | Perintah | Input / teacher | Output |
|---|---|---|---|---|
| 1 | DET-01 | `python -m src.training.train_detection --config configs/detection/teacher_x.yaml --set train.device=0` | dataset, `yolo26x.pt` | `outputs/detection/DET-01/` |
| 2 | DET-02 | `python -m src.training.train_detection --config configs/detection/baseline.yaml --set train.device=0` | `yolo26n.pt` | `outputs/detection/DET-02/` |
| 3 | DET-03 | `python -m src.training.train_detection_kd --config configs/detection/feature_kd.yaml --set train.device=0` | DET-01 | `outputs/detection/DET-03/` |
| 4 | DET-04 | `python -m src.training.train_detection_kd --config configs/detection/prediction_kd.yaml --set train.device=0` | DET-01 | `outputs/detection/DET-04/` |
| 5 | DET-05 | `python -m src.training.train_detection_kd --config configs/detection/feature_prediction_kd.yaml --set train.device=0` | DET-01 | `outputs/detection/DET-05/` |
| 6 | DET-06 | `python -m src.training.train_detection --config configs/detection/teacher_x_ca.yaml --set train.device=0` | `yolo26x.pt` (layer 0–22 + head ditransfer, CA acak) | `outputs/detection/DET-06/` |
| 7 | DET-07 | `python -m src.training.train_detection --config configs/detection/student_n_ca.yaml --set train.device=0` | `yolo26n.pt` (layer 0–22 + head ditransfer, CA acak) | `outputs/detection/DET-07/` |
| 8 | DET-08 | `python -m src.training.train_detection_kd --config configs/detection/ca_feature_kd.yaml --set train.device=0` | DET-06 | `outputs/detection/DET-08/` |
| 9 | DET-09 | `python -m src.training.train_detection_kd --config configs/detection/ca_prediction_kd.yaml --set train.device=0` | DET-06 | `outputs/detection/DET-09/` |
| 10 | DET-10 | `python -m src.training.train_detection_kd --config configs/detection/ca_feature_prediction_kd.yaml --set train.device=0` | DET-06 | `outputs/detection/DET-10/` |

Satu perintah untuk seluruh urutan: `bash scripts/train_all_detection.sh train.device=0`.

## 3. Monocular depth estimation

| Langkah | ID | Perintah | Input / teacher | Output |
|---|---|---|---|---|
| 1 | DEP-01 | `python -m src.training.train_depth --config configs/depth/teacher_x.yaml --set train.device=0` | `yolo26x-depth.pt` | `outputs/depth/DEP-01/` (terkalibrasi) |
| 2 | DEP-02 | `python -m src.training.train_depth --config configs/depth/baseline.yaml --set train.device=0` | `yolo26n-depth.pt` | `outputs/depth/DEP-02/` (terkalibrasi) |
| 3 | DEP-03 | `python -m src.training.train_depth_kd --config configs/depth/output_kd.yaml --set train.device=0` | DEP-01 | `outputs/depth/DEP-03/` (terkalibrasi) |
| 4 | DEP-04 | `python -m src.training.train_depth_kd --config configs/depth/feature_kd.yaml --set train.device=0` | DEP-01 | `outputs/depth/DEP-04/` (terkalibrasi) |
| 5 | DEP-05 | `python -m src.training.train_depth_kd --config configs/depth/boundary_kd.yaml --set train.device=0` | DEP-01 | `outputs/depth/DEP-05/` (terkalibrasi) |
| 6 | DEP-06 | `python -m src.training.train_depth_kd --config configs/depth/relative_kd.yaml --set train.device=0` | DEP-01 | `outputs/depth/DEP-06/` (terkalibrasi) |
| 7 | DEP-07 | `python -m src.training.train_depth_kd --config configs/depth/full_kd.yaml --set train.device=0` | DEP-01 | `outputs/depth/DEP-07/` (terkalibrasi) |
| 8 | DEP-08 | `python -m src.training.train_depth --config configs/depth/teacher_x_cbam.yaml --set train.device=0` | `yolo26x-depth.pt` (CBAM acak) | `outputs/depth/DEP-08/` (terkalibrasi) |
| 9 | DEP-09 | `python -m src.training.train_depth --config configs/depth/student_n_cbam.yaml --set train.device=0` | `yolo26n-depth.pt` (CBAM acak) | `outputs/depth/DEP-09/` (terkalibrasi) |
| 10 | DEP-10 | `python -m src.training.train_depth_kd --config configs/depth/cbam_output_kd.yaml --set train.device=0` | DEP-08 | `outputs/depth/DEP-10/` (terkalibrasi) |
| 11 | DEP-11 | `python -m src.training.train_depth_kd --config configs/depth/cbam_feature_kd.yaml --set train.device=0` | DEP-08 | `outputs/depth/DEP-11/` (terkalibrasi) |
| 12 | DEP-12 | `python -m src.training.train_depth_kd --config configs/depth/cbam_boundary_kd.yaml --set train.device=0` | DEP-08 | `outputs/depth/DEP-12/` (terkalibrasi) |
| 13 | DEP-13 | `python -m src.training.train_depth_kd --config configs/depth/cbam_relative_kd.yaml --set train.device=0` | DEP-08 | `outputs/depth/DEP-13/` (terkalibrasi) |
| 14 | DEP-14 | `python -m src.training.train_depth_kd --config configs/depth/cbam_full_kd.yaml --set train.device=0` | DEP-08 | `outputs/depth/DEP-14/` (terkalibrasi) |

Satu perintah: `bash scripts/train_all_depth.sh train.device=0`.

Setiap model depth dikalibrasi otomatis setelah training (log-affine pada split `val`). Kalibrasi ulang tanpa
training: `python -m src.training.train_depth --config <config> --calibrate-only`.

## 4. Isi folder eksperimen

`weights/best.pt` (dipilih dari fitness pada split `val`), `weights/last.pt`, `results.csv` (loss GT, setiap
komponen KD, metrik validasi per epoch), `results.png`, kurva PR/P/R/F1 dan confusion matrix (deteksi, dibuat
Ultralytics), `loss_curves.png`, `args.yaml` (argumen Ultralytics), `config_used.yaml` (config penelitian setelah
override), `model_*.yaml` (arsitektur yang dibangun), `metadata.json` (experiment ID, config, split + SHA-256,
seed, loss/KD dan bobotnya, kalibrasi, optimizer dan lr di `ultralytics_args`, versi Python/PyTorch/Ultralytics,
GPU, git commit, SHA-256 teacher), `gpu_memory.csv` (puncak memori GPU per epoch), `training_summary.json` (durasi,
path checkpoint, best fitness), dan untuk depth `calibration_{best,last}.json`.

## 5. Resume, pretrained, dan batas memori

- **Resume:** jalankan perintah yang sama dengan `--resume` (memakai `weights/last.pt` eksperimen itu).
- **Dari nol:** `--set model.weights=null` (terapkan pada seluruh eksperimen yang dibandingkan).
- **GPU tidak cukup:** turunkan `train.batch`. Trainer Ultralytics sudah mengakumulasi gradien sampai ukuran
  batch nominal `nbs` (default 64): `accumulate = round(nbs / batch)`, dan weight decay diskalakan
  `weight_decay · batch · accumulate / nbs` (`engine/trainer.py`). Jadi batch 8 → 8 langkah akumulasi, batch 16 →
  4, dengan batch efektif ≈ 64 pada keduanya. Perubahan batch tetap harus dicatat (memengaruhi statistik
  BatchNorm); `args.yaml` setiap run menyimpan nilainya.
- **Mixed precision:** `train.amp: true` (Tabel 3.7/3.8). Bila loss KD menjadi NaN pada AMP, coba
  `--set train.amp=false` dan catat.

## 6. Mengubah bobot / komponen KD

Bobot setiap komponen ada di YAML eksperimen (`kd.output.weight`, `kd.feature.weight`, `kd.boundary.weight`,
`kd.relative.weight`; deteksi: `kd.feature.weight` = `dis`, `kd.prediction.{weight,cls_weight,box_weight,temperature}`).
Nilai default berasal dari proposal (Pers. 3.10: 0,5/0,2/0,1/0,1; `dis = 6`) atau, untuk Prediction KD yang tidak
diberi nilai oleh proposal, 1,0 (keputusan K-1). Perubahan bobot harus dicatat sebagai eksperimen terpisah.
