# Analisis Knowledge Distillation pada YOLO26 untuk Deteksi Objek dan Estimasi Jarak Objek dalam Sistem Navigasi Otonom

Repository ini berisi seluruh kode penelitian tesis. Cakupannya mulai dari preprocessing SUN RGB-D, training
YOLO26/YOLO26-Depth beserta seluruh variasi Knowledge Distillation (KD), evaluasi, dan pemilihan model dengan
EWM-SAW, sampai integrasi deteksi objek dengan monocular depth estimation, estimasi jarak setiap objek, dan
pemilihan arah gerak berdasarkan ruang yang masih dapat dilalui pada antarmuka sistem.

> **Status hasil.** Kode telah diuji dengan unit test, integration test, dan smoke test end-to-end memakai data
> **sintetis** berformat SUN RGB-D di CPU (lihat [Hasil pengujian](#hasil-pengujian-kode)). **Belum ada training
> penuh** pada SUN RGB-D: dataset dan bobot pretrained tidak dapat diunduh dari lingkungan pengembangan, dan tidak
> ada GPU. Repository ini **tidak memuat angka performa model apa pun**. Seluruh angka penelitian harus berasal
> dari menjalankan pipeline di bawah ini pada perangkat penelitian.

---

## Daftar isi

1. [Informasi penelitian](#1-informasi-penelitian)
2. [Struktur repository](#2-struktur-repository)
3. [Persiapan lingkungan](#3-persiapan-lingkungan)
4. [Dataset: unduh, preprocessing, validasi](#4-dataset-sun-rgb-d)
5. [Training object detection](#5-training-object-detection)
6. [Training monocular depth estimation](#6-training-monocular-depth-estimation)
7. [Evaluasi dan pemilihan model](#7-evaluasi-dan-pemilihan-model)
8. [Integrasi model](#8-integrasi-model)
9. [Pengujian navigasi](#9-pengujian-navigasi)
10. [RUNNING THE COMPLETE RESEARCH PIPELINE](#10-running-the-complete-research-pipeline)
11. [Melanjutkan training dan memakai checkpoint](#11-melanjutkan-training-dan-memakai-checkpoint)
12. [Pengujian kode](#12-pengujian-kode)
13. [Batasan penelitian dan keputusan metodologis](#13-batasan-penelitian-dan-keputusan-metodologis)

---

## 1. Informasi penelitian

**Judul.** Analisis Knowledge Distillation pada YOLO26 untuk Deteksi Objek dan Estimasi Jarak Objek dalam Sistem
Navigasi Otonom.

**Tujuan** (proposal Sub-bab 1.3):

1. Menganalisis pengaruh KD dari YOLO26x terhadap kemampuan deteksi dan efisiensi komputasi YOLO26n, serta
   menentukan konfigurasi paling efisien dengan SAW berbobot EWM.
2. Menganalisis pengaruh KD dari YOLO26x-Depth terhadap estimasi kedalaman metrik dan efisiensi YOLO26n-Depth,
   serta menentukan konfigurasi paling efisien dengan SAW berbobot EWM.
3. Mengintegrasikan YOLO26n dan YOLO26n-Depth hasil KD untuk mengestimasi jarak setiap objek, lalu mengevaluasi
   akurasinya terhadap jarak aktual di lingkungan indoor.
4. Mengevaluasi ketepatan integrasi kedua model dalam menentukan arah gerak berdasarkan ruang yang masih dapat
   dilalui pada antarmuka sistem.

**Metodologi singkat.** Teacher berkapasitas besar (YOLO26x, YOLO26x-Depth) dilatih lebih dulu pada SUN RGB-D
dengan split resmi (5.285 train / 5.050 validation). Student YOLO26n dan YOLO26n-Depth kemudian dilatih tanpa KD
(baseline) dan dengan berbagai konfigurasi KD; teacher dibekukan, dan student tetap memperoleh supervisi ground
truth. Seluruh model dievaluasi dengan metrik akurasi dan efisiensi, lalu diperingkat dengan EWM-SAW. Student KD
terpilih diintegrasikan: satu citra RGB menghasilkan bounding box dan peta kedalaman metrik. Jarak objek dihitung
sebagai median depth pada inner ROI 60%, dan arah gerak dipilih dari lima sektor dalam ROI navigasi 60% di tengah
frame.

**Arsitektur:**

- YOLO26 (backbone C3k2/SPPF/C2PSA, neck FPN+PAN, Detect head anchor-free end-to-end).
- YOLO26-Depth (backbone dan neck yang sama dengan Depth head pada P3/P4/P5 dan log-depth).
- Varian teacher dengan Coordinate Attention (deteksi) atau CBAM (depth) pada P3/P4/P5 sebelum head.

Detail ada di [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md).

**Variasi eksperimen**

| Object detection | Monocular depth estimation |
|---|---|
| 5.1-1 YOLO26x (teacher) | 5.2-1 YOLO26x-Depth (teacher) |
| 5.1-2 YOLO26n baseline | 5.2-2 YOLO26n-Depth baseline |
| 5.1-3 YOLO26x → YOLO26n, Feature KD | 5.2-3 Depth Output Distillation |
| 5.1-4 YOLO26x → YOLO26n, Prediction KD | 5.2-4 Feature Distillation |
| 5.1-5 YOLO26x+CA → YOLO26n, KD (Feature KD) | 5.2-5 Boundary Distillation |
| 5.1-6 YOLO26x+CA → YOLO26n, Feature + Prediction KD | 5.2-6 Relative Depth Distillation |
| | 5.2-7 Full Depth KD (4 komponen) |
| | 5.2-8 YOLO26x-Depth+CBAM → YOLO26n-Depth, KD (Output) |
| | 5.2-9 YOLO26x-Depth+CBAM → YOLO26n-Depth, Full Depth KD |

Teacher YOLO26x+CA dan YOLO26x-Depth+CBAM dilatih sebagai eksperimen pendukung.

**Alur kerja sistem**

```
Citra RGB monocular
 ├─► YOLO26n (KD)       ─► bounding box + confidence (conf ≥ 0,25, NMS IoU 0,45, maks 100)
 └─► YOLO26n-Depth (KD) ─► depth map metrik (kalibrasi log-affine)
      └─► spatial alignment ─► inner ROI 60% ─► valid filter 0,1–10 m, R_valid ≥ 0,30 ─► clip p5–p95 ─► median
           └─► jarak per objek (axial) ─► ROI navigasi 60% ─► sektor FL | L | CTR | R | FR
                └─► blocked/free (jarak < 1,0 m, depth invalid, confidence < 0,20)
                     └─► prioritas CTR → L → R → FL → FR ─► FORWARD / TURN_LEFT / TURN_RIGHT / STOP
```

---

## 2. Struktur repository

```
navigasi-otonom-v1.0/
├── README.md, LICENSE (AGPL-3.0), requirements.txt, pyproject.toml, .gitignore
├── configs/      dataset/, detection/, depth/, evaluation/, system/  (seluruh hyperparameter dalam YAML)
├── datasets/     (tidak di-commit) raw/ = berkas SUN RGB-D asli, sunrgbd_yolo/ = hasil preprocessing
├── models/       arsitektur: CA, CBAM, builder YOLO26/YOLO26-Depth, wrapper teacher-student
├── src/          data/, losses/, training/, evaluation/, optimization/, integration/, navigation/, utils/
├── scripts/      skrip shell per tahap + scripts/data/ (validasi dataset, generator data sintetis)
├── app/          interface demonstrasi (Gradio / OpenCV)
├── tests/        unit test dan integration test (pytest)
├── docs/         metodologi, dataset, eksperimen, arsitektur, referensi, troubleshooting, audit
├── data_collection/  template CSV pengukuran jarak dan skenario navigasi (diisi peneliti)
└── outputs/      (tidak di-commit) checkpoint, log, evaluasi, laporan
```

Semua perintah dijalankan dari **root repository**. Perintah Python memakai bentuk `python -m paket.modul`.

### 2.1 Konfigurasi (YAML)

| File | Fungsi | Input | Output | Cara menjalankan |
|---|---|---|---|---|
| `configs/dataset/sunrgbd.yaml` | Path data mentah, split resmi, sumber depth/box, rentang depth, ekspektasi 5.285/5.050 | — | — | dibaca `src.data.prepare_sunrgbd`, `scripts/data/validate_dataset.py` |
| `configs/detection/_common.yaml` | Hyperparameter bersama Tabel 3.7, path data, integrity gate, parameter inferensi | — | — | diwarisi semua config deteksi (`_base_`) |
| `configs/detection/teacher_x.yaml` | 5.1-1 YOLO26x | `yolo26x.pt` | `outputs/detection/det_01_*` | `python -m src.training.train_detection --config ...` |
| `configs/detection/teacher_x_ca.yaml` | Teacher YOLO26x + CA | `yolo26x.pt` | `outputs/detection/det_00_*` | `train_detection` |
| `configs/detection/baseline.yaml` | 5.1-2 YOLO26n | `yolo26n.pt` | `det_02_*` | `train_detection` |
| `configs/detection/feature_kd.yaml` | 5.1-3 Feature KD | teacher `det_01` | `det_03_*` | `python -m src.training.train_detection_kd --config ...` |
| `configs/detection/prediction_kd.yaml` | 5.1-4 Prediction KD | teacher `det_01` | `det_04_*` | `train_detection_kd` |
| `configs/detection/ca_kd.yaml` | 5.1-5 teacher CA + Feature KD | teacher `det_00` | `det_05_*` | `train_detection_kd` |
| `configs/detection/ca_full_kd.yaml` | 5.1-6 teacher CA + Feature + Prediction KD | teacher `det_00` | `det_06_*` | `train_detection_kd` |
| `configs/depth/_common.yaml` | Hyperparameter bersama Tabel 3.8 + kalibrasi log-affine | — | — | diwarisi semua config depth |
| `configs/depth/teacher_x.yaml` | 5.2-1 YOLO26x-Depth | `yolo26x-depth.pt` | `outputs/depth/dep_01_*` | `python -m src.training.train_depth --config ...` |
| `configs/depth/teacher_x_cbam.yaml` | Teacher YOLO26x-Depth + CBAM | `yolo26x-depth.pt` | `dep_00_*` | `train_depth` |
| `configs/depth/baseline.yaml` | 5.2-2 YOLO26n-Depth | `yolo26n-depth.pt` | `dep_02_*` | `train_depth` |
| `configs/depth/output_kd.yaml` | 5.2-3 Output KD | teacher `dep_01` | `dep_03_*` | `python -m src.training.train_depth_kd --config ...` |
| `configs/depth/feature_kd.yaml` | 5.2-4 Feature KD | teacher `dep_01` | `dep_04_*` | `train_depth_kd` |
| `configs/depth/boundary_kd.yaml` | 5.2-5 Boundary KD | teacher `dep_01` | `dep_05_*` | `train_depth_kd` |
| `configs/depth/relative_kd.yaml` | 5.2-6 Relative KD | teacher `dep_01` | `dep_06_*` | `train_depth_kd` |
| `configs/depth/full_kd.yaml` | 5.2-7 Full Depth KD | teacher `dep_01` | `dep_07_*` | `train_depth_kd` |
| `configs/depth/cbam_kd.yaml` | 5.2-8 teacher CBAM + Output KD | teacher `dep_00` | `dep_08_*` | `train_depth_kd` |
| `configs/depth/cbam_full_kd.yaml` | 5.2-9 teacher CBAM + Full KD | teacher `dep_00` | `dep_09_*` | `train_depth_kd` |
| `configs/evaluation/detection.yaml` | Daftar model deteksi yang dievaluasi, titik operasi, benchmark | `best.pt` | `outputs/evaluation/detection` | `python -m src.evaluation.evaluate_detection` |
| `configs/evaluation/depth.yaml` | Daftar model depth, rentang depth, benchmark | `best.pt` | `outputs/evaluation/depth` | `python -m src.evaluation.evaluate_depth` |
| `configs/evaluation/selection.yaml` | Kriteria benefit/cost EWM-SAW per tugas | `*_results.csv` | `outputs/model_selection` | `python -m src.optimization.model_selection` |
| `configs/evaluation/distance.yaml` | Sumber jarak aktual (`sunrgbd` / `measured`) | sistem terintegrasi | `outputs/evaluation/distance` | `python -m src.evaluation.evaluate_distance` |
| `configs/evaluation/navigation.yaml` | Sumber arah sebenarnya (`scenarios` / `sunrgbd`) | sistem terintegrasi | `outputs/evaluation/navigation` | `python -m src.evaluation.evaluate_navigation` |
| `configs/system/inference.yaml` | Model terpilih, parameter deteksi, depth, jarak, fusion, navigasi | `selected_models.json` | `outputs/integrated_system` | dibaca integrasi, evaluasi, dan interface |

Setiap nilai dapat ditimpa dari CLI dengan `--set kunci.sub=nilai`, misalnya
`--set train.device=0 train.batch=8`.

### 2.2 Kode Python

| File | Fungsi | Input | Output | Cara menjalankan |
|---|---|---|---|---|
| `src/data/download_dataset.py` | Unduh SUNRGBD.zip, toolbox, Meta2DBB | URL resmi | `datasets/raw/` | `python -m src.data.download_dataset --out datasets/raw` |
| `src/data/prepare_depth.py` | Dekode bit-rotation, mm→m, valid mask, PNG uint16 lossless | PNG depth | array meter / PNG | modul (dipakai `prepare_sunrgbd`) |
| `src/data/pair_rgb_depth.py` | Penemuan scene, identitas scene, pairing dan pemeriksaan RGB-depth, intrinsics | folder SUNRGBD | `PairRecord` | modul |
| `src/data/prepare_annotations.py` | Pembacaan `SUNRGBDMeta2DBB_v2.mat` / poligon, pembersihan box, konversi YOLO | `.mat` / JSON | label `.txt` | modul |
| `src/data/prepare_sunrgbd.py` | Orkestrasi preprocessing dengan split resmi | `configs/dataset/sunrgbd.yaml` | `datasets/sunrgbd_yolo/` | `python -m src.data.prepare_sunrgbd` |
| `src/data/validate_dataset.py` | Integrity check dan gate training | dataset siap | `outputs/dataset/*` | `python scripts/data/validate_dataset.py` |
| `models/__init__.py` | Mendaftarkan CoordAtt/CBAM ke parser YAML Ultralytics | — | — | otomatis saat `import models` |
| `models/common/model_utils.py` | YAML ber-attention, build model, transfer bobot lintas layer sisipan, statistik parameter/FLOPs | YAML resmi | `nn.Module` | modul |
| `models/common/feature_hooks.py` | Forward hook untuk inspeksi layer tap KD | model | dict fitur | modul / test |
| `models/detection/coordinate_attention.py` | Coordinate Attention (Hou et al., 2021) | tensor | tensor | modul |
| `models/detection/yolo26_detector.py` | Menulis YAML model per eksperimen (resmi atau + attention) | config `model:` | `model_*.yaml` | modul |
| `models/detection/detection_teacher_student.py` | `DetectionKDModel`: Feature KD (Ultralytics) + Prediction KD | teacher, student | loss + log | dipakai `train_detection_kd` |
| `models/depth/cbam.py` | CBAM (Woo et al., 2018) | tensor | tensor | modul |
| `models/depth/yolo26_depth.py` | Inferensi depth, fitting dan penulisan kalibrasi log-affine | checkpoint | depth (m), `calibration_*.json` | modul / `train_depth --calibrate-only` |
| `models/depth/depth_teacher_student.py` | `DepthKDModel`: 4 komponen KD depth, proyektor teacher→student | teacher, student | loss + log | dipakai `train_depth_kd` |
| `src/losses/feature_loss.py` | Score-weighted L2 (deteksi), feature distillation multi-level (depth) | fitur | loss | modul |
| `src/losses/detection_kd.py` | Prediction KD kelas + box | output head | loss | modul |
| `src/losses/depth_output_loss.py` | Pers. 2.9 SmoothL1 log-depth bermask | depth | loss | modul |
| `src/losses/boundary_loss.py` | Pers. 2.11–2.12 gradient magnitude | depth | loss | modul |
| `src/losses/relative_depth_loss.py` | Pers. 2.13–2.14 ranking loss pasangan piksel | depth | loss | modul |
| `src/losses/combined_loss.py` | Total KD depth berbobot, toggle per komponen | depth, fitur | dict loss | modul |
| `src/training/trainer_utils.py` | Trainer mixin, substitusi kelas KD, metadata, gate, grafik loss | config | folder eksperimen | modul |
| `src/training/train_detection.py` | Training teacher / baseline deteksi | config | `outputs/detection/<id>` | `python -m src.training.train_detection --config ...` |
| `src/training/train_detection_kd.py` | Training student deteksi dengan KD | config + teacher | `outputs/detection/<id>` | `python -m src.training.train_detection_kd --config ...` |
| `src/training/train_depth.py` | Training teacher / baseline depth + kalibrasi log-affine | config | `outputs/depth/<id>` | `python -m src.training.train_depth --config ...` |
| `src/training/train_depth_kd.py` | Training student depth dengan KD + kalibrasi | config + teacher | `outputs/depth/<id>` | `python -m src.training.train_depth_kd --config ...` |
| `src/evaluation/metrics.py` | IoU, TP/FP/FN, P/R/F1, metrik depth, MAE/RMSE/MAPE, akurasi navigasi | prediksi + GT | dict | modul |
| `src/evaluation/benchmark.py` | Parameter, FLOPs, latensi, FPS, memori GPU; benchmark sistem | model / sistem | JSON | `python -m src.evaluation.benchmark --image ...` |
| `src/evaluation/evaluate_detection.py` | Evaluasi seluruh model deteksi | `best.pt` | `detection_results.csv/json` | `python -m src.evaluation.evaluate_detection` |
| `src/evaluation/evaluate_depth.py` | Evaluasi seluruh model depth | `best.pt` | `depth_results.csv/json` | `python -m src.evaluation.evaluate_depth` |
| `src/evaluation/evaluate_distance.py` | Akurasi jarak (same-box atau pengukuran) | sistem + GT | `distance_*.csv/json` | `python -m src.evaluation.evaluate_distance` |
| `src/evaluation/evaluate_navigation.py` | Akurasi keputusan arah | sistem + skenario | `navigation_*.csv/json` | `python -m src.evaluation.evaluate_navigation` |
| `src/optimization/entropy_weight.py` | EWM Pers. 3.29–3.32 | matriks keputusan | bobot | modul |
| `src/optimization/simple_additive_weighting.py` | SAW Pers. 3.27, 3.28, 3.33 | matriks + bobot | skor, ranking | modul |
| `src/optimization/model_selection.py` | EWM-SAW per tugas, pemilihan student KD | `*_results.csv` | `outputs/model_selection/*` | `python -m src.optimization.model_selection` |
| `src/integration/spatial_alignment.py` | Pers. 3.11–3.12 box ke grid depth | box | box | modul |
| `src/integration/inner_roi.py` | Pers. 3.13–3.17 inner ROI 60%, minimum 4 px | box | jendela piksel | modul |
| `src/integration/object_distance.py` | Pers. 3.18–3.19 jarak median, R_valid, axial/euclidean | depth + box | jarak | modul |
| `src/integration/detector_depth_fusion.py` | Pipeline lengkap satu frame + CLI pada citra/folder | citra | visualisasi, CSV, JSON | `python -m src.integration.detector_depth_fusion --source ...` |
| `src/navigation/navigation_roi.py` | ROI navigasi 60% tengah | W, H | ROI | modul |
| `src/navigation/sector_division.py` | 5 sektor lebar 0,12 W | ROI | sektor | modul |
| `src/navigation/obstacle_analysis.py` | Overlap ratio (Pers. 3.44), aturan blocking | objek + sektor | status sektor | modul |
| `src/navigation/direction_decision.py` | Prioritas CTR→L→R→FL→FR | status sektor | arah | modul |
| `src/utils/config.py` | Pemuat YAML (`_base_`), override CLI, path relatif root | YAML | dict | modul |
| `src/utils/seed.py` | Seed dan determinisme | — | — | modul |
| `src/utils/logger.py` | Logging konsol/berkas | — | — | modul |
| `src/utils/io.py` | Penulisan JSON/CSV/YAML | data | berkas | modul |
| `src/utils/checkpoint.py` | Lokasi best/last, baca/tulis checkpoint | config | path | modul |
| `src/utils/visualization.py` | Colormap depth (hanya tampilan), box, sektor | citra | citra | modul |
| `src/utils/report.py` | Menyusun `RESEARCH_REPORT.md` dari artefak nyata | `outputs/` | Markdown | `python -m src.utils.report --root outputs` |
| `app/main.py` | Interface Gradio (gambar + webcam) / jendela OpenCV | citra/webcam | tampilan | `python -m app.main` |
| `app/components/panels.py` | Panel visual dan tabel interface | hasil pipeline | gambar/tabel | modul |
| `scripts/data/validate_dataset.py` | Wrapper CLI validasi dataset | config | laporan | `python scripts/data/validate_dataset.py` |
| `scripts/data/make_synthetic_sunrgbd.py` | Data **sintetis** berformat SUN RGB-D untuk smoke test | — | `outputs/smoke*/raw` | `python scripts/data/make_synthetic_sunrgbd.py` |
| `tests/*.py` | Unit test dan integration test | — | — | `python -m pytest` |

### 2.3 Skrip shell

| File | Fungsi | Cara menjalankan |
|---|---|---|
| `scripts/prepare_dataset.sh` | (unduh) → prepare → validate | `bash scripts/prepare_dataset.sh [--download]` |
| `scripts/train_all_detection.sh` | 7 run deteksi, berurutan sesuai dependensi teacher | `bash scripts/train_all_detection.sh [override ...]` |
| `scripts/train_all_depth.sh` | 10 run depth | `bash scripts/train_all_depth.sh [override ...]` |
| `scripts/evaluate_all.sh` | Evaluasi deteksi dan depth (termasuk teacher ber-attention) | `bash scripts/evaluate_all.sh` |
| `scripts/select_models.sh` | EWM-SAW kedua tugas | `bash scripts/select_models.sh` |
| `scripts/run_integrated_system.sh` | Integrasi, jarak, navigasi, benchmark sistem | `bash scripts/run_integrated_system.sh <folder_citra> [citra_benchmark]` |
| `scripts/smoke_test.sh` | Seluruh pipeline dengan data sintetis (CPU, sekitar 3 menit) | `bash scripts/smoke_test.sh` |

### 2.4 Dokumentasi

| File | Isi |
|---|---|
| `docs/METHODOLOGY.md` | Pemetaan persamaan proposal ke kode dan **seluruh keputusan implementasi** (beberapa perlu konfirmasi) |
| `docs/REFERENCES_AND_IMPLEMENTATION.md` | Referensi jurnal, repository, file sumber, adaptasi, status |
| `docs/DATASET.md` | Struktur dan pemrosesan SUN RGB-D |
| `docs/EXPERIMENTS.md` | Matriks eksperimen, struktur output, protokol pengukuran jarak dan skenario navigasi |
| `docs/MODEL_ARCHITECTURE.md` | Arsitektur YOLO26, YOLO26-Depth, sisipan CA/CBAM, teacher-student |
| `docs/TROUBLESHOOTING.md` | Masalah umum dan solusinya |
| `docs/AUDIT.md` | Audit kondisi awal repository, repository referensi, dan Ultralytics |

---

## 3. Persiapan lingkungan

Lingkungan penelitian (Tabel 3.5):

- Python 3.11.
- PyTorch ≥ 2.4 dan torchvision ≥ 0.19.
- Ultralytics ≥ 8.4.194 dan < 9. Versi 8.4.194 adalah versi yang diverifikasi; ia berisi
  `ultralytics/nn/distill_model.py` dan task `depth`.
- GPU NVIDIA dengan CUDA (penelitian: RTX 5070, driver 590.48.01, CUDA 13.1).

```bash
git clone https://github.com/aauliariza/navigasi-otonom-v1.0.git
```

```bash
cd navigasi-otonom-v1.0
```

```bash
python3.11 -m venv .venv
```

```bash
source .venv/bin/activate
```

(Windows: `.venv\Scripts\activate`.)

Instal PyTorch sesuai versi CUDA dari https://pytorch.org/get-started/locally/. Contoh untuk CUDA 12.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Instal library lainnya (Ultralytics, OpenCV, SciPy, PyYAML, Pandas, Matplotlib, Gradio, pytest):

```bash
pip install -r requirements.txt
```

Periksa GPU dan versi:

```bash
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', ultralytics.__version__)"
```

Periksa instalasi dengan unit test (tanpa dataset):

```bash
python -m pytest -q
```

Bobot pretrained resmi (`yolo26n.pt`, `yolo26x.pt`, `yolo26n-depth.pt`, `yolo26x-depth.pt`) diunduh otomatis oleh
Ultralytics saat training pertama. Untuk melatih dari nol, gunakan `--set model.weights=null` (lihat keputusan H-2).

---

## 4. Dataset SUN RGB-D

Detail lengkap ada di [docs/DATASET.md](docs/DATASET.md).

1. **Unduh** (sekitar 6,4 GB) dari https://rgbd.cs.princeton.edu/:

   ```bash
   python -m src.data.download_dataset --out datasets/raw
   ```

   Atau unduh manual `SUNRGBD.zip`, `SUNRGBDtoolbox.zip`, dan `SUNRGBDMeta2DBB_v2.mat`.
2. **Penempatan.** `datasets/raw/SUNRGBD/` (berisi `kv1 kv2 realsense xtion`),
   `datasets/raw/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat`, dan `datasets/raw/SUNRGBDMeta2DBB_v2.mat`. Path
   lain dapat diatur di `configs/dataset/sunrgbd.yaml`.
3. **Pairing, dekode depth, konversi meter, valid mask, dan anotasi YOLO** dijalankan sekaligus:

   ```bash
   python -m src.data.prepare_sunrgbd --config configs/dataset/sunrgbd.yaml
   ```

   - Pairing berdasarkan identitas scene, dengan pemeriksaan resolusi dan berkas rusak.
   - `D_mm = (D_raw >> 3) | (D_raw << 13)`, lalu `D_m = D_mm / 1000`.
   - Valid bila 0,001 ≤ D ≤ 10 m; selain itu bernilai 0.
   - Box dibersihkan (di luar citra, tidak valid, sisi < 4 px) dan dikonversi ke format YOLO kelas 0 `objek`.
4. **Validasi** (wajib lulus sebelum training):

   ```bash
   python scripts/data/validate_dataset.py --config configs/dataset/sunrgbd.yaml
   ```

   Hasilnya: `outputs/dataset/integrity_report.json` (`"passed": true`), `integrity_issues.csv`,
   `objects_per_image.{csv,png}`, dan `samples/`.

Alternatif satu langkah: `bash scripts/prepare_dataset.sh`.

---

## 5. Training object detection

Urutan wajib: teacher dilatih sebelum student KD. Path teacher diambil otomatis dari `kd.teacher_config`.
Tambahkan `--set train.device=0` untuk memilih GPU.

1. YOLO26x (teacher):

   ```bash
   python -m src.training.train_detection --config configs/detection/teacher_x.yaml
   ```

2. YOLO26n baseline:

   ```bash
   python -m src.training.train_detection --config configs/detection/baseline.yaml
   ```

3. YOLO26n dengan Feature KD:

   ```bash
   python -m src.training.train_detection_kd --config configs/detection/feature_kd.yaml
   ```

4. YOLO26n dengan Prediction KD:

   ```bash
   python -m src.training.train_detection_kd --config configs/detection/prediction_kd.yaml
   ```

5. Teacher YOLO26x + Coordinate Attention, lalu YOLO26n dengan KD dari teacher tersebut:

   ```bash
   python -m src.training.train_detection --config configs/detection/teacher_x_ca.yaml
   ```

   ```bash
   python -m src.training.train_detection_kd --config configs/detection/ca_kd.yaml
   ```

6. YOLO26n dengan CA teacher + Feature KD + Prediction KD:

   ```bash
   python -m src.training.train_detection_kd --config configs/detection/ca_full_kd.yaml
   ```

Atau semuanya sekaligus: `bash scripts/train_all_detection.sh train.device=0`.

Setiap komponen loss tercatat per epoch di `outputs/detection/<id>/results.csv`:

- `train/box_loss`, `cls_loss`, `l1_loss`: ground truth.
- `train/dis_loss`: Feature KD.
- `train/pkd_loss`, `pkd_cls`, `pkd_box`: Prediction KD.

Grafiknya ada di `loss_curves.png`.

---

## 6. Training monocular depth estimation

Setelah training, setiap model depth dikalibrasi log-affine pada split train dan hasilnya ditulis ke `best.pt`
(lihat `calibration_best.json`).

1. YOLO26x-Depth:

   ```bash
   python -m src.training.train_depth --config configs/depth/teacher_x.yaml
   ```

2. YOLO26n-Depth baseline:

   ```bash
   python -m src.training.train_depth --config configs/depth/baseline.yaml
   ```

3. Depth Output Distillation:

   ```bash
   python -m src.training.train_depth_kd --config configs/depth/output_kd.yaml
   ```

4. Feature Distillation:

   ```bash
   python -m src.training.train_depth_kd --config configs/depth/feature_kd.yaml
   ```

5. Boundary Distillation:

   ```bash
   python -m src.training.train_depth_kd --config configs/depth/boundary_kd.yaml
   ```

6. Relative Depth Distillation:

   ```bash
   python -m src.training.train_depth_kd --config configs/depth/relative_kd.yaml
   ```

7. Full Depth KD:

   ```bash
   python -m src.training.train_depth_kd --config configs/depth/full_kd.yaml
   ```

8. Teacher YOLO26x-Depth + CBAM, lalu CBAM + KD:

   ```bash
   python -m src.training.train_depth --config configs/depth/teacher_x_cbam.yaml
   ```

   ```bash
   python -m src.training.train_depth_kd --config configs/depth/cbam_kd.yaml
   ```

9. CBAM + Full Depth KD:

   ```bash
   python -m src.training.train_depth_kd --config configs/depth/cbam_full_kd.yaml
   ```

Atau semuanya sekaligus: `bash scripts/train_all_depth.sh train.device=0`.

Kolom loss di `results.csv`:

- `train/dlog_loss`, `dgrad_loss`: ground truth.
- `train/kd_output`, `kd_feature`, `kd_boundary`, `kd_relative`: kontribusi KD terbobot.

Bobot KD dapat diubah, misalnya `--set kd.feature.weight=0.3`.

---

## 7. Evaluasi dan pemilihan model

- **Evaluasi deteksi:**

  ```bash
  python -m src.evaluation.evaluate_detection --config configs/evaluation/detection.yaml --include-extra
  ```

  Menghasilkan mAP50, mAP50-95, P/R/F1 (conf 0,25, IoU 0,5), FPS, parameter, GFLOPs, dan memori GPU ke
  `outputs/evaluation/detection/detection_results.{csv,json}`.
- **Evaluasi depth:**

  ```bash
  python -m src.evaluation.evaluate_depth --config configs/evaluation/depth.yaml --include-extra
  ```

  Menghasilkan δ1–3, AbsRel, SqRel, RMSE, RMSE log, FPS, parameter, GFLOPs, dan memori GPU ke
  `outputs/evaluation/depth/depth_results.{csv,json}`.
- **Efisiensi komputasi.** Bagian dari kedua evaluasi di atas (benchmark model-only, B = 1, 50 warm-up, 300
  iterasi). Perangkat dan perangkat lunak dicatat di `hardware.json`.
- **EWM dan SAW** (dihitung terpisah untuk deteksi dan depth):

  ```bash
  python -m src.optimization.model_selection --config configs/evaluation/selection.yaml
  ```

  Untuk setiap tugas, `outputs/model_selection/<tugas>/` berisi:

  - `decision_matrix.csv`: nilai metrik.
  - `ewm_proportions.csv` dan `ewm_weights.csv`: proporsi, entropy, derajat diversifikasi, bobot.
  - `normalized_matrix.csv`: nilai normalisasi.
  - `saw_ranking.csv`: kontribusi `w·r`, nilai SAW, ranking.
  - `selection_report.{json,md}`: model terpilih.
- **Checkpoint terpilih** untuk integrasi ada di `outputs/model_selection/selected_models.json`, berisi student KD
  berperingkat tertinggi per tugas (keputusan M-3).

---

## 8. Integrasi model

Sistem membaca `selected_models.json`. Checkpoint juga dapat ditentukan manual dengan
`--set models.detection_weights=... models.depth_weights=...`.

```bash
python -m src.integration.detector_depth_fusion --config configs/system/inference.yaml --source path/ke/citra_atau_folder
```

Untuk setiap citra, prosesnya:

1. Object detection YOLO26n (conf 0,25, IoU 0,45, maksimum 100).
2. Depth estimation YOLO26n-Depth (depth metrik).
3. Spatial alignment box ke grid depth (Pers. 3.11–3.12).
4. Inner ROI 60%.
5. Filter depth valid 0,1–10 m dan **valid ratio** (R_valid ≥ 0,30).
6. Clip persentil 5–95, lalu median (jarak axial).
7. Visualisasi.

Hasil di `outputs/integrated_system/`:

- `<nama>_result.jpg`: box, confidence, jarak, sektor, arah, dan depth map.
- `objects.csv`: bbox, confidence, jarak, status valid, R_valid.
- `decisions.csv`: status sektor dan arah.
- `results.json`.

Untuk memakai jarak Euclidean (opsional), beri `--set distance.mode=euclidean "distance.intrinsics=[[fx,0,cx],[0,fy,cy],[0,0,1]]"`.

Akurasi jarak:

```bash
python -m src.evaluation.evaluate_distance --config configs/evaluation/distance.yaml
```

- `gt_source: sunrgbd` (default): protokol same-box pada depth ground truth split val.
- `gt_source: measured`: jarak hasil pengukuran langsung (protokol di [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)).

Keluarannya `distance_per_object.csv` (jarak sebenarnya, estimasi, selisih absolut, error %) dan
`distance_summary.json` (MAE, RMSE, MAPE, jumlah objek valid dan invalid).

---

## 9. Pengujian navigasi

- **Input gambar:** `python -m src.integration.detector_depth_fusion --source <citra>` atau tab "Gambar RGB" di
  interface.
- **Webcam** (satu kamera RGB monocular):

  ```bash
  python -m app.main --config configs/system/inference.yaml
  ```

  Buka http://127.0.0.1:7860, tab "Webcam". Untuk jendela OpenCV lokal:

  ```bash
  python -m app.main --config configs/system/inference.yaml --mode opencv --camera 0
  ```

- **ROI navigasi global:** 0,20 W ≤ x < 0,80 W dengan tinggi penuh (`navigation.roi_width_ratio`).
- **Lima sektor:** FL, L, CTR, R, FR, masing-masing selebar 0,12 W.
- **Blocked/free.** Objek menempati sektor bila overlap ≥ 0,05. Sektor *blocked* bila ada objek dengan jarak
  < 1,0 m, jarak invalid (`invalid_depth_blocks`), atau confidence fusion < 0,20 (`low_confidence_blocks`). Semua
  ambang dapat diatur di `configs/system/inference.yaml`.
- **Arah gerak:** sektor free pertama menurut prioritas CTR → L → R → FL → FR menentukan FORWARD (CTR),
  TURN_LEFT (L/FL), atau TURN_RIGHT (R/FR). Bila semua sektor blocked, hasilnya STOP.
- **Akurasi navigasi:**

  ```bash
  python -m src.evaluation.evaluate_navigation --config configs/evaluation/navigation.yaml
  ```

  Skenario diisi di `data_collection/navigation_scenarios.csv`. Hasilnya: `navigation_per_scenario.csv` (status
  setiap sektor, arah prediksi, arah sebenarnya, benar/salah), `confusion_matrix.csv`, dan
  `navigation_summary.json` (akurasi %).

Sistem hanya menampilkan keputusan. Tidak ada kendali motor, kursi roda, ROS, atau path planning.

---

## 10. RUNNING THE COMPLETE RESEARCH PIPELINE

Jalankan dari root repository dengan virtual environment aktif. Tambahkan `--set train.device=0` (training) atau
`--set device=0` (evaluasi) bila perlu. Sebelum memakai data asli, Anda dapat memastikan semua tahap berjalan
dengan `bash scripts/smoke_test.sh` (data sintetis, sekitar 3 menit di CPU).

| # | Tahap | Tujuan | Input | File yang dijalankan | Perintah | Output | Lokasi output | Berikutnya |
|---|---|---|---|---|---|---|---|---|
| 1 | Setup environment | Menyiapkan Python 3.11, PyTorch CUDA, dependensi | — | `requirements.txt` | lihat §3, lalu `python -m pytest -q` | venv siap, test lulus | `.venv/` | 2 |
| 2 | Persiapan dataset | Split resmi, pairing, depth meter, label YOLO | `datasets/raw/*` | `src/data/prepare_sunrgbd.py` | `python -m src.data.prepare_sunrgbd --config configs/dataset/sunrgbd.yaml` | images/labels/depth, manifest, YAML dataset | `datasets/sunrgbd_yolo/` | 3 |
| 3 | Validasi dataset | Integrity gate | dataset siap | `scripts/data/validate_dataset.py` | `python scripts/data/validate_dataset.py` | `integrity_report.json` (`passed`) | `outputs/dataset/` | 4 (hanya bila lulus) |
| 4 | Training YOLO26x | Teacher deteksi | dataset deteksi | `src/training/train_detection.py` | `python -m src.training.train_detection --config configs/detection/teacher_x.yaml` | best/last.pt, results.csv | `outputs/detection/det_01_yolo26x_teacher/` | 5 |
| 5 | Training YOLO26n baseline | Student tanpa KD | dataset deteksi | `train_detection.py` | `python -m src.training.train_detection --config configs/detection/baseline.yaml` | idem | `outputs/detection/det_02_yolo26n_baseline/` | 6 |
| 6 | Training seluruh eksperimen Detection KD | 5.1-3 … 5.1-6 (+ teacher CA) | teacher det_01 / det_00 | `train_detection_kd.py`, `train_detection.py` | `bash scripts/train_all_detection.sh` menjalankan 4–6 berurutan; perintah per eksperimen ada di §5 langkah 3–6 | best.pt student, log KD | `outputs/detection/det_0{0,3,4,5,6}_*` | 7 |
| 7 | Evaluasi seluruh model deteksi | Metrik + efisiensi | best.pt | `src/evaluation/evaluate_detection.py` | `python -m src.evaluation.evaluate_detection --config configs/evaluation/detection.yaml --include-extra` | detection_results.csv/json | `outputs/evaluation/detection/` | 8 |
| 8 | Training YOLO26x-Depth | Teacher depth + kalibrasi | dataset depth | `src/training/train_depth.py` | `python -m src.training.train_depth --config configs/depth/teacher_x.yaml` | best.pt terkalibrasi | `outputs/depth/dep_01_yolo26x_depth_teacher/` | 9 |
| 9 | Training YOLO26n-Depth baseline | Student depth tanpa KD | dataset depth | `train_depth.py` | `python -m src.training.train_depth --config configs/depth/baseline.yaml` | idem | `outputs/depth/dep_02_*` | 10 |
| 10 | Training seluruh eksperimen Depth KD | 5.2-3 … 5.2-9 (+ teacher CBAM) | teacher dep_01 / dep_00 | `train_depth_kd.py`, `train_depth.py` | `bash scripts/train_all_depth.sh` menjalankan 8–10 berurutan; perintah per eksperimen ada di §6 langkah 3–9 | best.pt student terkalibrasi | `outputs/depth/dep_0{0,3..9}_*` | 11 |
| 11 | Evaluasi seluruh model depth | Metrik + efisiensi | best.pt | `src/evaluation/evaluate_depth.py` | `python -m src.evaluation.evaluate_depth --config configs/evaluation/depth.yaml --include-extra` | depth_results.csv/json, sampel | `outputs/evaluation/depth/` | 12 |
| 12 | Perhitungan EWM dan SAW | Bobot objektif + ranking | *_results.csv | `src/optimization/model_selection.py` | `python -m src.optimization.model_selection --config configs/evaluation/selection.yaml` | bobot, normalisasi, SAW, ranking | `outputs/model_selection/{detection,depth}/` | 13 |
| 13 | Pemilihan model student | Student KD terbaik per tugas | selection_report | (hasil tahap 12) | `cat outputs/model_selection/selected_models.json` | path checkpoint terpilih | `outputs/model_selection/selected_models.json` | 14 |
| 14 | Integrasi kedua model | Deteksi + depth + jarak + arah | citra RGB, model terpilih | `src/integration/detector_depth_fusion.py` | `python -m src.integration.detector_depth_fusion --source datasets/sunrgbd_yolo/images/val` | visualisasi, objects.csv, decisions.csv | `outputs/integrated_system/` | 15 |
| 15 | Evaluasi object distance | MAE, RMSE, MAPE | sistem + depth GT / pengukuran | `src/evaluation/evaluate_distance.py` | `python -m src.evaluation.evaluate_distance --config configs/evaluation/distance.yaml` (lalu `--set gt_source=measured` untuk data lapangan) | distance_per_object.csv, summary | `outputs/evaluation/distance/` | 16 |
| 16 | Pengujian navigasi | Keputusan arah per skenario | citra skenario indoor | `src/integration/detector_depth_fusion.py`, `app/main.py` | `python -m src.integration.detector_depth_fusion --source data_collection/` | status sektor + arah per skenario | `outputs/integrated_system/` | 17 |
| 17 | Evaluasi akurasi navigasi | Akurasi (Pers. 3.45) | `navigation_scenarios.csv` | `src/evaluation/evaluate_navigation.py` | `python -m src.evaluation.evaluate_navigation --config configs/evaluation/navigation.yaml` | per-skenario, confusion matrix, akurasi | `outputs/evaluation/navigation/` | 18 |
| 18 | Menjalankan interface | Demonstrasi gambar + webcam | model terpilih | `app/main.py` | `python -m app.main --config configs/system/inference.yaml` | UI di http://127.0.0.1:7860 | — | 19 |
| 19 | Laporan hasil penelitian | Rangkuman seluruh artefak + benchmark sistem | `outputs/` | `src/evaluation/benchmark.py`, `src/utils/report.py` | `python -m src.evaluation.benchmark --image <citra_uji>` lalu `python -m src.utils.report --root outputs --out outputs/RESEARCH_REPORT.md` | benchmark_system.json, RESEARCH_REPORT.md | `outputs/` | selesai |

Catatan per tahap:

- Tahap 3 menghentikan pipeline bila gagal. Semua script training memeriksa `integrity_report.json`.
- Tahap 6 dan 10 bergantung pada teacher tahap 4/8 (atau teacher attention). Script akan berhenti dengan pesan
  jelas bila teacher belum ada.
- Tahap 15 dan 17 dengan data lapangan memerlukan berkas `data_collection/*.csv` yang diisi dari pengukuran
  nyata.
- Benchmark sistem (tahap 19) mengukur tiga hal terpisah: model-only (di tahap 7/11), pipeline dua model tanpa
  pre/post-processing, dan end-to-end per frame termasuk preprocessing dan postprocessing.

---

## 11. Melanjutkan training dan memakai checkpoint

- **Resume** setelah proses terhenti. Perintahnya sama dengan menambah `--resume`; training dilanjutkan dari
  `weights/last.pt` beserta optimizer, scheduler, EMA, dan epoch:

  ```bash
  python -m src.training.train_detection_kd --config configs/detection/ca_full_kd.yaml --resume
  ```

  Untuk run KD, teacher dibangun ulang dari path yang tersimpan di checkpoint.
- **Checkpoint terbaik/terakhir:** `outputs/<detection|depth>/<experiment_id>/weights/{best,last}.pt`.
  `best.pt` milik student KD hanya berisi model student (tanpa teacher dan proyektor).
- **Memakai checkpoint yang sudah ada sebagai teacher** tanpa melatih ulang:
  `--set kd.teacher_weights=path/ke/best.pt`.
- **Kalibrasi ulang depth** tanpa training:

  ```bash
  python -m src.training.train_depth --config configs/depth/baseline.yaml --calibrate-only
  ```

- **Inferensi satu model** (Ultralytics API, setelah `import models`):

  ```bash
  python -c "import models; from ultralytics import YOLO; r=YOLO('outputs/detection/det_03_yolo26n_feature_kd/weights/best.pt').predict('img.jpg', conf=0.25, iou=0.45); print(r[0].boxes)"
  ```

---

## 12. Pengujian kode

```bash
python -m pytest -q
```

```bash
bash scripts/smoke_test.sh
```

<a id="hasil-pengujian-kode"></a>**Hasil pengujian pada lingkungan pengembangan** (CPU Intel Xeon, Python 3.11.18,
torch 2.15.0, ultralytics 8.4.194, tanpa GPU):

- `pytest`: **83 test lulus**.
  - 78 unit test: dataset, CA/CBAM, KD deteksi, KD depth, metrik, EWM/SAW, jarak, navigasi.
  - 5 integration test: gate dataset, training teacher/baseline/KD, interrupt lalu resume run KD, kalibrasi,
    evaluasi, EWM-SAW, integrasi, evaluasi jarak dan navigasi (termasuk mode `measured` dan `scenarios`).
- `scripts/smoke_test.sh`: **lulus**. Mencakup 17 run training (7 deteksi, 10 depth, 1 epoch, imgsz 96, teacher
  memakai arsitektur n agar cepat) hingga evaluasi, EWM-SAW, integrasi, evaluasi jarak dan navigasi, benchmark,
  dan laporan. Durasinya sekitar 3 menit.
- `ruff check .` dan `python -m compileall` bersih. Semua modul dapat di-import tanpa circular import.

**Belum dapat diuji di lingkungan pengembangan:**

- Pembacaan berkas SUN RGB-D asli (`allsplit.mat`, `SUNRGBDMeta2DBB_v2.mat`, PNG `depth_bfx`). Formatnya ditiru
  oleh data sintetis, tetapi berkas asli tidak dapat diunduh (domain diblokir). Jalankan tahap 2–3 dan periksa
  laporannya sebelum training.
- Training penuh 100 epoch, CUDA/AMP, bobot pretrained resmi, dan pengukuran memori GPU.
- Webcam fisik dan interaksi Gradio di browser. Pembangunan UI dan seluruh panel sudah diuji tanpa tampilan.

---

## 13. Batasan penelitian dan keputusan metodologis

Batasan proposal yang dijaga oleh kode:

- Lingkungan indoor, objek statis, pencahayaan relatif stabil.
- Satu citra RGB sebagai satu-satunya input saat inferensi; tidak ada sensor depth, LiDAR, atau stereo.
- SUN RGB-D dengan split resmi.
- Satu kelas `objek`.
- Hanya keluarga YOLO26.
- Konfigurasi KD sesuai daftar eksperimen, tanpa Object ROI Distillation.
- Evaluasi dengan metrik proposal.
- Pemilihan model dengan EWM-SAW.
- Navigasi hanya berbasis ruang bebas pada antarmuka, tanpa kendali fisik kursi roda, embedded deployment, path
  planning, atau SLAM.

Beberapa hal tidak ditetapkan proposal atau tampak keliru ketik. Semuanya diselesaikan dengan default yang dapat
dikonfigurasi dan **didokumentasikan di [docs/METHODOLOGY.md](docs/METHODOLOGY.md)**. Yang perlu dikonfirmasi
pembimbing atau peneliti:

- **A-1.** CA dan CBAM tidak tercantum dalam proposal. Penempatannya di P3/P4/P5 sebelum head.
- **A-2.** Jenis KD eksperimen 5.1-5 adalah Feature KD; eksperimen 5.2-8 adalah Output KD.
- **D-1.** Sumber depth: `depth_bfx`.
- **C-1.** Kalibrasi log-affine di-fit pada split train, bukan validation.
- **K-1.** Rumus dan bobot Prediction KD.
- **E-3.** Definisi kriteria memori GPU.
- **N-1.** Definisi "confidence fusion".
- **H-2.** Inisialisasi dari bobot pretrained resmi.
- **M-2.** Koreksi rumus normalisasi cost SAW (Pers. 3.28).
- **I-1.** Koreksi Pers. 3.17.

## Lisensi

AGPL-3.0, mengikuti Ultralytics yang dipakai dan diperluas (lihat `LICENSE` dan
[docs/REFERENCES_AND_IMPLEMENTATION.md](docs/REFERENCES_AND_IMPLEMENTATION.md)).
