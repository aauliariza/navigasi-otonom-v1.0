# Panduan Training

Semua perintah dijalankan dari root repository setelah dataset lulus validasi
(`outputs/dataset/integrity_report.json` berisi `"passed": true`; training menolak berjalan bila tidak).
GPU dipilih dengan `--set train.device=0`. Setiap nilai YAML dapat ditimpa dengan `--set kunci.sub=nilai`.

## 1. Urutan dan ketergantungan

Teacher harus selesai lebih dulu: script KD membaca `outputs/<task>/<teacher-ID>/weights/best.pt`, menolak
berjalan bila berkas itu tidak ada, dan mencatat SHA-256-nya di `metadata.json` student.

```
DET-01 ──► DET-03, DET-04            TEACHER-CA ──► DET-05, DET-06            DET-02 (mandiri)
DEP-01 ──► DEP-03 … DEP-07           TEACHER-CBAM ──► DEP-08, DEP-09          DEP-02 (mandiri)
```

## 2. Deteksi objek

| Langkah | ID | Perintah | Input | Output utama |
|---|---|---|---|---|
| 1 | DET-01 | `python -m src.training.train_detection --config configs/detection/teacher_x.yaml --set train.device=0` | dataset, `yolo26x.pt` | `outputs/detection/DET-01/weights/best.pt` |
| 2 | — | `python -m src.evaluation.evaluate_detection --set "models=[{name: DET-01 YOLO26x, config: configs/detection/teacher_x.yaml}]" device=0` | DET-01 `best.pt` | `outputs/evaluation/detection/val/` |
| 3 | DET-02 | `python -m src.training.train_detection --config configs/detection/baseline.yaml --set train.device=0` | `yolo26n.pt` | `outputs/detection/DET-02/` |
| 4 | DET-03 | `python -m src.training.train_detection_kd --config configs/detection/feature_kd.yaml --set train.device=0` | DET-01 | `outputs/detection/DET-03/` |
| 5 | DET-04 | `python -m src.training.train_detection_kd --config configs/detection/prediction_kd.yaml --set train.device=0` | DET-01 | `outputs/detection/DET-04/` |
| 6 | TEACHER-CA | `python -m src.training.train_detection --config configs/detection/teacher_x_ca.yaml --set train.device=0` | `yolo26x.pt` (layer 0–22 + head ditransfer, CA acak) | `outputs/detection/TEACHER-CA/` |
| 7 | DET-05 | `python -m src.training.train_detection_kd --config configs/detection/ca_kd.yaml --set train.device=0` | TEACHER-CA | `outputs/detection/DET-05/` |
| 8 | DET-06 | `python -m src.training.train_detection_kd --config configs/detection/ca_full_kd.yaml --set train.device=0` | TEACHER-CA | `outputs/detection/DET-06/` |

Satu perintah untuk seluruh urutan: `bash scripts/train_all_detection.sh train.device=0`.

## 3. Monocular depth estimation

| Langkah | ID | Perintah | Input | Output utama |
|---|---|---|---|---|
| 1 | DEP-01 | `python -m src.training.train_depth --config configs/depth/teacher_x.yaml --set train.device=0` | `yolo26x-depth.pt` | `outputs/depth/DEP-01/weights/best.pt` (terkalibrasi) |
| 2 | — | `python -m src.evaluation.evaluate_depth --set "models=[{name: DEP-01 YOLO26x-Depth, config: configs/depth/teacher_x.yaml}]" device=0` | DEP-01 | `outputs/evaluation/depth/val/` |
| 3 | DEP-02 | `python -m src.training.train_depth --config configs/depth/baseline.yaml --set train.device=0` | `yolo26n-depth.pt` | `outputs/depth/DEP-02/` |
| 4 | DEP-03 | `python -m src.training.train_depth_kd --config configs/depth/output_kd.yaml --set train.device=0` | DEP-01 | `outputs/depth/DEP-03/` |
| 5 | DEP-04 | `python -m src.training.train_depth_kd --config configs/depth/feature_kd.yaml --set train.device=0` | DEP-01 | `outputs/depth/DEP-04/` |
| 6 | DEP-05 | `python -m src.training.train_depth_kd --config configs/depth/boundary_kd.yaml --set train.device=0` | DEP-01 | `outputs/depth/DEP-05/` |
| 7 | DEP-06 | `python -m src.training.train_depth_kd --config configs/depth/relative_kd.yaml --set train.device=0` | DEP-01 | `outputs/depth/DEP-06/` |
| 8 | DEP-07 | `python -m src.training.train_depth_kd --config configs/depth/full_kd.yaml --set train.device=0` | DEP-01 | `outputs/depth/DEP-07/` |
| 9 | TEACHER-CBAM | `python -m src.training.train_depth --config configs/depth/teacher_x_cbam.yaml --set train.device=0` | `yolo26x-depth.pt` | `outputs/depth/TEACHER-CBAM/` |
| 10 | DEP-08 | `python -m src.training.train_depth_kd --config configs/depth/cbam_kd.yaml --set train.device=0` | TEACHER-CBAM | `outputs/depth/DEP-08/` |
| 11 | DEP-09 | `python -m src.training.train_depth_kd --config configs/depth/cbam_full_kd.yaml --set train.device=0` | TEACHER-CBAM | `outputs/depth/DEP-09/` |

Satu perintah: `bash scripts/train_all_depth.sh train.device=0`.

Setiap model depth dikalibrasi otomatis setelah training (log-affine pada validasi internal). Kalibrasi ulang tanpa
training: `python -m src.training.train_depth --config <config> --calibrate-only`.

## 4. Isi folder eksperimen

`weights/best.pt` (dipilih dari fitness validasi internal), `weights/last.pt`, `results.csv` (loss GT, setiap
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
