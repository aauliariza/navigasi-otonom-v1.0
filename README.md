# Analisis Knowledge Distillation pada YOLO26 untuk Deteksi Objek dan Estimasi Jarak Objek dalam Sistem Navigasi Otonom

Repository kode penelitian tesis: preprocessing SUN RGB-D, training YOLO26 / YOLO26-Depth dengan seluruh variasi
Knowledge Distillation (KD), evaluasi akurasi dan efisiensi, pemilihan model dengan Entropy Weight Method (EWM) dan
Simple Additive Weighting (SAW), integrasi deteksi objek dengan monocular metric depth estimation, estimasi jarak
setiap objek, pemilihan arah gerak berdasarkan ruang yang masih dapat dilalui, serta antarmuka pengujian dengan
gambar, video, dan webcam.

> **Status hasil.** Kode telah diuji dengan unit test, integration test, dan smoke test end-to-end memakai data
> **sintetis** berformat SUN RGB-D di CPU ([§13](#13-pengujian-kode-dan-status-verifikasi)). **Belum ada training
> penuh** pada SUN RGB-D: dataset, bobot pretrained, GPU, dan kamera tidak tersedia di lingkungan pengembangan.
> Repository ini **tidak memuat angka performa model apa pun**; laporan menulis `NOT MEASURED` untuk metrik yang
> belum diukur. Seluruh angka penelitian harus berasal dari menjalankan pipeline ini pada perangkat penelitian.

## Daftar isi

1. [Informasi penelitian](#1-informasi-penelitian)
2. [Struktur repository dan fungsi setiap file](#2-struktur-repository-dan-fungsi-setiap-file)
3. [Instalasi environment](#3-instalasi-environment)
4. [Persiapan dataset](#4-persiapan-dataset)
5. [Training deteksi objek](#5-training-deteksi-objek)
6. [Training monocular depth estimation](#6-training-monocular-depth-estimation)
7. [Evaluasi dan pemilihan model](#7-evaluasi-dan-pemilihan-model)
8. [Integrasi dan estimasi jarak](#8-integrasi-dan-estimasi-jarak)
9. [Navigasi](#9-navigasi)
10. [Testing: gambar, video, webcam, antarmuka](#10-testing-gambar-video-webcam-antarmuka)
11. [Menjalankan seluruh penelitian](#11-menjalankan-seluruh-penelitian)
12. [Troubleshooting](#12-troubleshooting)
13. [Pengujian kode dan status verifikasi](#13-pengujian-kode-dan-status-verifikasi)
14. [Batasan dan keputusan metodologis](#14-batasan-dan-keputusan-metodologis)

---

## 1. Informasi penelitian

**Tujuan** (proposal Sub-bab 1.3):

1. Menganalisis pengaruh KD dari YOLO26x terhadap deteksi objek dan efisiensi YOLO26n, serta menentukan konfigurasi
   paling efisien dengan SAW berbobot EWM.
2. Menganalisis pengaruh KD dari YOLO26x-Depth terhadap estimasi kedalaman metrik dan efisiensi YOLO26n-Depth, serta
   menentukan konfigurasi paling efisien dengan SAW berbobot EWM.
3. Mengintegrasikan YOLO26n dan YOLO26n-Depth hasil KD untuk mengestimasi jarak setiap objek dan mengevaluasi
   akurasinya terhadap jarak sebenarnya di lingkungan indoor.
4. Mengevaluasi ketepatan integrasi kedua model dalam menentukan arah gerak berdasarkan ruang yang masih dapat dilalui
   pada antarmuka sistem.

**Gambaran sistem.** Satu citra RGB dari satu kamera monocular masuk ke dua model: YOLO26n (bounding box satu kelas
`objek`) dan YOLO26n-Depth (peta kedalaman metrik). Box dipetakan ke grid depth, jarak tiap objek = median depth
valid pada inner ROI 60%, lalu ROI navigasi 60% tengah frame dibagi lima sektor yang dinilai free/blocked, dan arah
FORWARD / TURN_LEFT / TURN_RIGHT / STOP direkomendasikan pada antarmuka. **Tidak ada kendali motor/kursi roda.**

**Arsitektur penelitian**

```
SUN RGB-D ─► preprocessing (pairing, depth meter, label YOLO, split train 5.285 / val 5.050)
   ├─► DETEKSI: DET-01 YOLO26x teacher ─► DET-02 baseline, DET-03 Feature KD, DET-04 Prediction KD
   │            TEACHER-CA (YOLO26x + Coordinate Attention) ─► DET-05 Prediction KD, DET-06 Feature + Prediction KD
   └─► DEPTH:   DEP-01 YOLO26x-Depth teacher ─► DEP-02 baseline, DEP-03 Output, DEP-04 Feature, DEP-05 Boundary,
                                               DEP-06 Relative, DEP-07 Full (4 komponen)
                TEACHER-CBAM (YOLO26x-Depth + CBAM) ─► DEP-08 Output KD, DEP-09 Full KD
evaluasi (val 5.050) ─► EWM + SAW per tugas ─► model terbaik (student KD)
integrasi ─► alignment ─► inner ROI ─► filter valid ─► persentil 5–95 ─► median = jarak ─► ROI navigasi ─► 5 sektor
          ─► free/blocked ─► arah ─► antarmuka (gambar / video / webcam)
```

**Daftar eksperimen** (rincian: [docs/EXPERIMENT_MATRIX.md](docs/EXPERIMENT_MATRIX.md))

| ID | Teacher | Metode | Student | | ID | Teacher | Metode KD | Student |
|---|---|---|---|---|---|---|---|---|
| DET-01 | YOLO26x | Tanpa KD | — | | DEP-01 | YOLO26x-Depth | Tanpa KD | — |
| DET-02 | — | Baseline | YOLO26n | | DEP-02 | — | Baseline | YOLO26n-Depth |
| DET-03 | YOLO26x | Feature KD | YOLO26n | | DEP-03 | YOLO26x-Depth | Depth Output | YOLO26n-Depth |
| DET-04 | YOLO26x | Prediction KD | YOLO26n | | DEP-04 | YOLO26x-Depth | Feature | YOLO26n-Depth |
| DET-05 | YOLO26x + CA | Prediction KD | YOLO26n | | DEP-05 | YOLO26x-Depth | Boundary | YOLO26n-Depth |
| DET-06 | YOLO26x + CA | Feature + Prediction KD | YOLO26n | | DEP-06 | YOLO26x-Depth | Relative Depth | YOLO26n-Depth |
| | | | | | DEP-07 | YOLO26x-Depth | Output + Feature + Boundary + Relative | YOLO26n-Depth |
| | | | | | DEP-08 | YOLO26x-Depth + CBAM | Depth Output | YOLO26n-Depth |
| | | | | | DEP-09 | YOLO26x-Depth + CBAM | Output + Feature + Boundary + Relative | YOLO26n-Depth |

Teacher ber-attention dilatih sebagai run pendukung `TEACHER-CA` dan `TEACHER-CBAM`. Object ROI Distillation
dari draf BAB 3 **tidak** dipakai (keputusan X-2).

**Teknologi:** Python 3.11 (diuji juga 3.13), PyTorch ≥ 2.4, Ultralytics 8.4.194 (YOLO26, task `depth`,
`DistillationModel`), OpenCV, NumPy, SciPy, PyYAML, Pandas, Matplotlib, ultralytics-thop (FLOPs), Gradio
(antarmuka), pytest.

---

## 2. Struktur repository dan fungsi setiap file

```
navigasi-otonom-v1.0/
├── README.md, LICENSE (AGPL-3.0), requirements.txt, environment.yml, pyproject.toml, .gitignore
├── configs/          dataset/, detection/, depth/, evaluation/, system/  (seluruh hyperparameter dan threshold)
├── src/              data/, losses/, training/, evaluation/, optimization/, integration/, navigation/, inference/, utils/
├── models/           CA, CBAM, builder YOLO26/YOLO26-Depth, wrapper teacher–student
├── app/              antarmuka Gradio
├── scripts/          skrip shell per tahap, data sintetis
├── tests/            pytest (unit + integration)
├── docs/             dokumentasi metodologi, keputusan, provenance, panduan
├── data_collection/  template CSV pengukuran jarak dan skenario navigasi (diisi peneliti)
├── datasets/         (tidak di-commit) raw/ = SUN RGB-D asli, sunrgbd_yolo/ = hasil preprocessing
└── outputs/          (tidak di-commit) checkpoint, log, evaluasi, seleksi, inferensi, laporan
```

Alasan struktur `src/` dan pemetaannya ke struktur yang disarankan prompt: [docs/README_ARCHITECTURE.md](docs/README_ARCHITECTURE.md).
Semua perintah dijalankan dari **root repository**; nilai YAML dapat ditimpa dengan `--set kunci.sub=nilai`.

### 2.1 Kode Python

| File | Fungsi (sesuai implementasi) |
|---|---|
| `src/data/download_dataset.py` | Mengunduh `SUNRGBD.zip`, `SUNRGBDtoolbox.zip`, `SUNRGBDMeta2DBB_v2.mat` dari situs resmi lalu mengekstrak ke `datasets/raw/` |
| `src/data/pair_rgb_depth.py` | Menemukan folder scene, menurunkan identitas scene portabel (`kv1/NYUdata/NYU0001`), memasangkan `image/*.jpg` dengan `depth_bfx/*.png` dari folder scene yang sama, memeriksa keberadaan/keunikan/resolusi/rasio depth valid, membaca intrinsics |
| `src/data/prepare_depth.py` | Dekode bit-rotation `(D>>3)\|(D<<13)`, mm→m, valid mask 0,001–10 m (di luar rentang = 0/invalid), simpan/muat PNG uint16 mm lossless |
| `src/data/prepare_annotations.py` | Membaca `SUNRGBDMeta2DBB_v2.mat` (atau poligon `annotation2Dfinal`), clip box ke citra, buang box invalid/sisi < 4 px, konversi YOLO kelas 0 (Pers. 3.3–3.8) |
| `src/data/create_split.py` | Mode split opsional `internal`: menurunkan grup rekaman tiap scene dan membagi `alltrain` menjadi `train` + validasi internal tanpa berbagi grup (tidak dipakai pada mode default `proposal`) |
| `src/data/prepare_sunrgbd.py` | Orkestrasi preprocessing: split (`proposal`: train 5.285 / val 5.050; atau `internal`), pairing, depth meter, label YOLO, `manifest.csv`, `splits.json`, YAML dataset deteksi/depth, laporan |
| `src/data/validate_dataset.py` | Integrity check (pairing, resolusi, rentang depth, label, kebocoran sampel dan grup, jumlah resmi) → `integrity_report.json`; `require_dataset_ok` menjadi gate semua training |
| `models/__init__.py` | Mendaftarkan `CoordAtt`/`CBAM` ke parser YAML Ultralytics agar model ber-attention dapat dibangun dan dimuat |
| `models/common/model_utils.py` | Membangun YAML ber-attention (CA/CBAM pada P3/P4/P5), membangun model, transfer bobot pretrained melintasi layer sisipan, menghitung parameter dan FLOPs (`thop`, error eksplisit) |
| `models/common/feature_hooks.py` | Forward hook untuk memeriksa layer tap KD (input head) |
| `models/detection/coordinate_attention.py` | Coordinate Attention (Hou et al., 2021): pooling arah H dan W, transformasi bersama, atensi `a_h`, `a_w` |
| `models/detection/yolo26_detector.py` | Menulis YAML model per eksperimen (YAML resmi atau varian ber-attention) ke folder eksperimen |
| `models/detection/detection_teacher_student.py` | `DetectionKDModel` (turunan `DistillationModel` Ultralytics): teacher frozen, Feature KD score-weighted L2 (`dis`), Prediction KD, log per komponen |
| `models/depth/cbam.py` | CBAM (Woo et al., 2018): channel gate (MLP avg+max) lalu spatial gate (konv 7×7 + BN) |
| `models/depth/yolo26_depth.py` | Preprocessing dan inferensi depth, fitting kalibrasi log-affine (a, b) pada split validasi dan penulisan ke checkpoint |
| `models/depth/depth_teacher_student.py` | `DepthKDModel`: tap input Depth head, proyektor teacher→student, loss GT + 4 komponen KD |
| `src/losses/feature_loss.py` | Score-weighted L2 (deteksi) dan feature distillation multi-level dengan proyektor + normalisasi L2 (depth) |
| `src/losses/detection_kd.py` | Prediction KD: soft-label BCE kelas + L1 box berbobot confidence teacher, cabang one2many dan one2one |
| `src/losses/depth_output_loss.py` | Depth output distillation: SmoothL1 pada log-depth dengan mask valid (Pers. 2.9) |
| `src/losses/boundary_loss.py` | Boundary distillation: gradien Gx, Gy, magnitudo, SmoothL1 dengan mask gradien valid (Pers. 2.11–2.12) |
| `src/losses/relative_depth_loss.py` | Relative depth distillation: 4.096 pasangan/citra (sampler ber-seed), toleransi 0,03, ranking loss (Pers. 2.13–2.14) |
| `src/losses/combined_loss.py` | Total KD depth berbobot (0,5/0,2/0,1/0,1), toggle per komponen dari YAML |
| `src/training/trainer_utils.py` | Mixin trainer (model ber-attention, transfer bobot, log memori GPU, grafik loss), substitusi kelas KD, metadata eksperimen (split SHA-256, versi, hardware), gate dataset |
| `src/training/train_detection.py` | CLI training tanpa KD: DET-01, DET-02, TEACHER-CA (`--resume` didukung) |
| `src/training/train_detection_kd.py` | CLI training student deteksi dengan KD: DET-03…DET-06 |
| `src/training/train_depth.py` | CLI training depth tanpa KD (DEP-01, DEP-02, TEACHER-CBAM) + kalibrasi log-affine pada `val`; `--calibrate-only` |
| `src/training/train_depth_kd.py` | CLI training student depth dengan KD: DEP-03…DEP-09 (+ kalibrasi) |
| `src/evaluation/metrics.py` | IoU, pencocokan TP/FP/FN, P/R/F1, akumulator metrik depth (δ1–3, AbsRel, SqRel, RMSE, RMSE log), MAE/RMSE/MAPE, akurasi navigasi |
| `src/evaluation/benchmark.py` | FPS/latensi model-only, pipeline, end-to-end; memori GPU puncak; deskripsi hardware; CLI benchmark sistem |
| `src/evaluation/evaluate_detection.py` | Evaluasi semua model deteksi pada `val`/`test`: mAP, P/R/F1 titik operasi, FPS, parameter, FLOPs, memori → CSV/JSON |
| `src/evaluation/evaluate_depth.py` | Evaluasi semua model depth: metrik, panel RGB/GT/prediksi/error, `.npy`, efisiensi → CSV/JSON |
| `src/evaluation/evaluate_distance.py` | Akurasi jarak per objek (same-box SUN RGB-D atau pengukuran lapangan): MAE, RMSE, MAPE, grafik |
| `src/evaluation/evaluate_navigation.py` | Akurasi keputusan arah (skenario indoor atau referensi SUN RGB-D), confusion matrix CSV/PNG |
| `src/optimization/entropy_weight.py` | EWM (Pers. 3.29–3.32) dengan penanganan nol, konstan, negatif, NaN/Inf, m = 1 |
| `src/optimization/simple_additive_weighting.py` | Normalisasi SAW benefit/cost, skor, ranking |
| `src/optimization/model_selection.py` | EWM-SAW terpisah untuk deteksi dan depth, tolak hasil `test`, ranking + grafik, `selected_models.json` |
| `src/integration/spatial_alignment.py` | Pemetaan box dan intrinsics dari koordinat citra ke grid depth (Pers. 3.11–3.12) |
| `src/integration/inner_roi.py` | Inner ROI 60% dengan margin 20%, ukuran minimum 4 px (Pers. 3.13–3.17) |
| `src/integration/object_distance.py` | Filter valid 0,1–10 m, R_valid ≥ 0,30, clip persentil 5–95, median; mode axial (utama) / euclidean |
| `src/integration/detector_depth_fusion.py` | `PerceptionPipeline` (pemuatan model + error informatif, deteksi, depth, jarak, navigasi, render) dan CLI folder citra |
| `src/navigation/navigation_roi.py` | ROI navigasi global 60% tengah, tinggi penuh |
| `src/navigation/sector_division.py` | Lima sektor sama lebar FL, L, CTR, R, FR |
| `src/navigation/obstacle_analysis.py` | Overlap ratio (Pers. 3.44), aturan blocking (jarak < 1 m, depth invalid, confidence fusion < 0,20) |
| `src/navigation/direction_decision.py` | Prioritas CTR→L→R→FL→FR, pemetaan ke FORWARD/TURN_LEFT/TURN_RIGHT/STOP |
| `src/inference/common.py` | Argumen CLI bersama, pemuatan pipeline, baris CSV hasil, pengukur FPS |
| `src/inference/predict_image.py` | Testing gambar (berkas/folder JPG/PNG/BMP/TIFF/WEBP) → visualisasi, CSV, JSON, depth `.npy` opsional |
| `src/inference/predict_video.py` | Testing video per frame → video hasil (box, confidence, jarak, ROI, sektor, arah, FPS, depth map), CSV per frame |
| `src/inference/webcam.py` | Testing realtime satu webcam RGB: status kamera, FPS, simpan/rekam, penanganan kamera hilang/frame kosong/resolusi |
| `src/utils/config.py` | Pemuat YAML dengan `_base_`, override `--set`, path relatif root |
| `src/utils/seed.py` | Seed dan mode deterministik |
| `src/utils/logger.py` | Logger konsol/berkas |
| `src/utils/io.py` | Tulis/baca JSON, CSV, YAML |
| `src/utils/checkpoint.py` | Folder eksperimen per ID, path best/last, baca/tulis checkpoint Ultralytics |
| `src/utils/visualization.py` | Colormap depth, peta error, box, sektor, teks status, grafik jarak dan confusion matrix (hanya tampilan) |
| `src/utils/report.py` | Laporan gabungan per experiment ID (val/test) dari artefak nyata, `NOT MEASURED` bila belum diukur |
| `app/main.py` | Antarmuka Gradio: tab gambar, video, webcam; info model; tombol "Simpan hasil"; mode jendela OpenCV |
| `app/components/panels.py` | Panel deteksi/depth, tabel objek dan sektor, teks keputusan |
| `scripts/data/validate_dataset.py` | Wrapper CLI `src/data/validate_dataset.py` |
| `scripts/data/make_synthetic_sunrgbd.py` | Data **sintetis** berformat SUN RGB-D untuk smoke test (bukan data penelitian) |
| `tests/conftest.py`, `tests/scripts_helpers.py` | Fixture model kecil dan dataset sintetis |
| `tests/test_*.py` | Unit dan integration test (rincian di §13) |

### 2.2 Konfigurasi, skrip, dokumentasi

| File | Fungsi |
|---|---|
| `configs/dataset/sunrgbd.yaml` | Path data mentah, sumber depth/box, rentang depth, mode split (`proposal` default), jumlah resmi, opsi validasi internal |
| `configs/detection/_common.yaml` | Hyperparameter Tabel 3.7, augmentasi deteksi, parameter inferensi; diwarisi semua config deteksi |
| `configs/detection/{teacher_x,baseline,feature_kd,prediction_kd,teacher_x_ca,ca_kd,ca_full_kd}.yaml` | DET-01, DET-02, DET-03, DET-04, TEACHER-CA, DET-05, DET-06 |
| `configs/depth/_common.yaml` | Hyperparameter Tabel 3.8, augmentasi depth, kalibrasi log-affine (split `val`) |
| `configs/depth/{teacher_x,baseline,output_kd,feature_kd,boundary_kd,relative_kd,full_kd,teacher_x_cbam,cbam_kd,cbam_full_kd}.yaml` | DEP-01…DEP-07, TEACHER-CBAM, DEP-08, DEP-09 |
| `configs/evaluation/detection.yaml`, `depth.yaml` | Daftar model, split, titik operasi/mode head, benchmark |
| `configs/evaluation/selection.yaml` | Kriteria benefit/cost EWM-SAW, kebijakan data hilang, larangan split test |
| `configs/evaluation/distance.yaml`, `navigation.yaml` | Sumber ground truth jarak/arah, split, output |
| `configs/system/inference.yaml` | Model terpilih, deteksi (conf, IoU, max_det, mode head), depth, jarak, fusion, aturan navigasi |
| `scripts/prepare_dataset.sh` | (unduh) → prepare → validate |
| `scripts/train_all_detection.sh`, `train_all_depth.sh` | Seluruh training sesuai urutan dependensi teacher |
| `scripts/evaluate_all.sh` | Evaluasi deteksi dan depth pada `val` lalu `test` |
| `scripts/select_models.sh` | EWM-SAW kedua tugas |
| `scripts/run_integrated_system.sh` | Testing gambar, evaluasi jarak & navigasi, benchmark, laporan |
| `scripts/smoke_test.sh` | Seluruh pipeline pada data sintetis (CPU, ±4 menit) |
| `docs/README_ARCHITECTURE.md` | Arsitektur repository, model, teacher ber-attention, teacher–student, pemetaan struktur |
| `docs/EXPERIMENT_MATRIX.md` | Matriks eksperimen, penggunaan split, struktur output, protokol data lapangan |
| `docs/EXPERIMENT_DESIGN_DECISIONS.md` | Ketidaksesuaian antar-sumber dan seluruh keputusan implementasi |
| `docs/METHODOLOGY_MAPPING.md` | Pemetaan sub-bab/persamaan proposal dan bagian prompt ke kode |
| `docs/SOURCE_PROVENANCE.md` | Paper, repository, berkas yang diadaptasi, modifikasi, status verifikasi |
| `docs/DATASET_PREPARATION.md`, `TRAINING_GUIDE.md`, `EVALUATION_GUIDE.md`, `TROUBLESHOOTING.md` | Panduan per tahap |
| `docs/AUDIT.md` | Audit repository (dua sesi), repository referensi, dan Ultralytics |

---

## 3. Instalasi environment

Lingkungan penelitian (Tabel 3.5): Python 3.11, PyTorch ≥ 2.4, torchvision ≥ 0.19, Ultralytics 8.4.x < 9 (versi
terverifikasi: **8.4.194**, yang memuat task `depth` dan `nn/distill_model.py`), GPU NVIDIA (RTX 5070, CUDA 13.1).

1. Clone repository:

   ```bash
   git clone https://github.com/aauliariza/navigasi-otonom-v1.0.git
   ```

   ```bash
   cd navigasi-otonom-v1.0
   ```

2. Virtual environment (atau `conda env create -f environment.yml` lalu `conda activate navigasi-otonom`):

   ```bash
   python3.11 -m venv .venv
   ```

   ```bash
   source .venv/bin/activate
   ```

   Windows: `.venv\Scripts\activate`.

3. PyTorch sesuai CUDA (https://pytorch.org/get-started/locally/), contoh CUDA 12.8:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

   CPU saja: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

4. Dependensi lain:

   ```bash
   pip install -r requirements.txt
   ```

   **Jangan** memasang paket `thop` (lama); ia menimpa `ultralytics-thop` dan membuat FLOPs gagal dihitung.

5. Verifikasi instalasi dan GPU:

   ```bash
   python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

6. Unit test awal (tanpa dataset, ±1 menit di CPU):

   ```bash
   python -m pytest tests/ -v
   ```

Bobot pretrained resmi (`yolo26n.pt`, `yolo26x.pt`, `yolo26n-depth.pt`, `yolo26x-depth.pt`) diunduh otomatis oleh
Ultralytics saat training pertama; untuk training dari awal gunakan `--set model.weights=null` (keputusan H-2).

---

## 4. Persiapan dataset

Rincian: [docs/DATASET_PREPARATION.md](docs/DATASET_PREPARATION.md).

1. **Unduh** SUN RGB-D (±6,4 GB) dari https://rgbd.cs.princeton.edu/:

   ```bash
   python -m src.data.download_dataset --out datasets/raw
   ```

2. **Ekstrak** (dilakukan otomatis oleh langkah 1; bila mengunduh manual, ekstrak `SUNRGBD.zip` dan
   `SUNRGBDtoolbox.zip` ke `datasets/raw/` dan letakkan `SUNRGBDMeta2DBB_v2.mat` di `datasets/raw/`).
3. **Direktori dataset**: `datasets/raw/SUNRGBD/{kv1,kv2,realsense,xtion}`,
   `datasets/raw/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat`, `datasets/raw/SUNRGBDMeta2DBB_v2.mat`. Path lain:
   ubah `source.*` di `configs/dataset/sunrgbd.yaml`.
4–9. **Pairing RGB-depth, decoding depth, konversi meter, dataset deteksi, anotasi YOLO, split resmi + validasi
   internal** dijalankan oleh satu perintah:

   ```bash
   python -m src.data.prepare_sunrgbd --config configs/dataset/sunrgbd.yaml
   ```

   - Pairing berdasarkan identitas scene (bukan urutan berkas), dengan pemeriksaan resolusi dan berkas rusak.
   - `D_mm = (D_raw >> 3) | (D_raw << 13)`, `D_m = D_mm / 1000`; valid bila 0,001 ≤ D ≤ 10 m, selain itu 0.
   - Box: clip ke citra, buang invalid/sisi < 4 px, YOLO kelas 0 `objek`.
   - Split (`split.mode: proposal`, sesuai Tabel 3.6, sama untuk deteksi dan depth): `alltrain` → `train`
     (5.285), `alltest` → `val` (5.050). Tidak ada split test terpisah; uji independen sistem memakai data indoor
     yang Anda ambil sendiri (§8–§10). Mode opsional `--set split.mode=internal` membuat validasi internal dari
     `alltrain` dan menjadikan 5.050 data sebagai `test`.
10–12. **Pemeriksaan kebocoran, validasi dataset, laporan preprocessing**:

   ```bash
   python scripts/data/validate_dataset.py --config configs/dataset/sunrgbd.yaml
   ```

   Lulus bila `outputs/dataset/integrity_report.json` berisi `"passed": true` (termasuk
   `split_mode: proposal`, train 5.285, val 5.050). Laporan preprocessing: `datasets/sunrgbd_yolo/meta/{preparation_report.json,
   manifest.csv,splits.json,preparation_failures.csv}`.

Alternatif satu langkah: `bash scripts/prepare_dataset.sh` (tambahkan `--download` untuk mengunduh).

---

## 5. Training deteksi objek

Rincian input/output per langkah: [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md). Tambahkan `--set train.device=0`
untuk GPU. Setiap eksperimen menulis ke `outputs/detection/<ID>/` (`weights/best.pt`, `weights/last.pt`,
`results.csv`, kurva, `metadata.json`, dll.).

1. YOLO26x teacher (DET-01):

   ```bash
   python -m src.training.train_detection --config configs/detection/teacher_x.yaml
   ```

2. Evaluasi teacher (split val):

   ```bash
   python -m src.evaluation.evaluate_detection --set "models=[{name: DET-01 YOLO26x, config: configs/detection/teacher_x.yaml}]"
   ```

3. YOLO26n baseline (DET-02):

   ```bash
   python -m src.training.train_detection --config configs/detection/baseline.yaml
   ```

4. DET-03 Feature KD (membaca `outputs/detection/DET-01/weights/best.pt`):

   ```bash
   python -m src.training.train_detection_kd --config configs/detection/feature_kd.yaml
   ```

5. DET-04 Prediction KD:

   ```bash
   python -m src.training.train_detection_kd --config configs/detection/prediction_kd.yaml
   ```

6. Teacher YOLO26x + Coordinate Attention (TEACHER-CA):

   ```bash
   python -m src.training.train_detection --config configs/detection/teacher_x_ca.yaml
   ```

7. DET-05 (teacher CA, Prediction KD):

   ```bash
   python -m src.training.train_detection_kd --config configs/detection/ca_kd.yaml
   ```

8. DET-06 (teacher CA, Feature + Prediction KD):

   ```bash
   python -m src.training.train_detection_kd --config configs/detection/ca_full_kd.yaml
   ```

9. Evaluasi seluruh eksperimen: [§7](#7-evaluasi-dan-pemilihan-model).

Semua langkah training sekaligus: `bash scripts/train_all_detection.sh train.device=0`. Resume: tambahkan `--resume`.

---

## 6. Training monocular depth estimation

Setiap model depth dikalibrasi log-affine pada split `val` setelah training; checkpoint di
`outputs/depth/<ID>/weights/best.pt` sudah menghasilkan depth metrik.

1. YOLO26x-Depth teacher (DEP-01):

   ```bash
   python -m src.training.train_depth --config configs/depth/teacher_x.yaml
   ```

2. Evaluasi teacher:

   ```bash
   python -m src.evaluation.evaluate_depth --set "models=[{name: DEP-01 YOLO26x-Depth, config: configs/depth/teacher_x.yaml}]"
   ```

3. YOLO26n-Depth baseline (DEP-02):

   ```bash
   python -m src.training.train_depth --config configs/depth/baseline.yaml
   ```

4. DEP-03 Depth Output Distillation:

   ```bash
   python -m src.training.train_depth_kd --config configs/depth/output_kd.yaml
   ```

5. DEP-04 Feature Distillation:

   ```bash
   python -m src.training.train_depth_kd --config configs/depth/feature_kd.yaml
   ```

6. DEP-05 Boundary Distillation:

   ```bash
   python -m src.training.train_depth_kd --config configs/depth/boundary_kd.yaml
   ```

7. DEP-06 Relative Depth Distillation:

   ```bash
   python -m src.training.train_depth_kd --config configs/depth/relative_kd.yaml
   ```

8. DEP-07 Full Depth KD:

   ```bash
   python -m src.training.train_depth_kd --config configs/depth/full_kd.yaml
   ```

9. Teacher YOLO26x-Depth + CBAM (TEACHER-CBAM):

   ```bash
   python -m src.training.train_depth --config configs/depth/teacher_x_cbam.yaml
   ```

10. DEP-08 (teacher CBAM, Depth Output KD):

    ```bash
    python -m src.training.train_depth_kd --config configs/depth/cbam_kd.yaml
    ```

11. DEP-09 (teacher CBAM, Full Depth KD):

    ```bash
    python -m src.training.train_depth_kd --config configs/depth/cbam_full_kd.yaml
    ```

12. Evaluasi seluruh eksperimen: [§7](#7-evaluasi-dan-pemilihan-model).

Semua sekaligus: `bash scripts/train_all_depth.sh train.device=0`.

---

## 7. Evaluasi dan pemilihan model

Rincian: [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md).

1–6. **Evaluasi deteksi dan depth** (akurasi, FPS, parameter, FLOPs, memori GPU) pada split `val` (5.050):

   ```bash
   python -m src.evaluation.evaluate_detection --config configs/evaluation/detection.yaml --include-extra --set device=0
   ```

   ```bash
   python -m src.evaluation.evaluate_depth --config configs/evaluation/depth.yaml --include-extra --set device=0
   ```

   Hasil: `outputs/evaluation/{detection,depth}/val/*_results.{csv,json}`. FPS diukur model-only (B = 1, 640 px,
   50 warm-up + 300 iterasi); parameter dan FLOPs dihitung pada model ter-*fuse*; memori GPU = puncak
   `max_memory_allocated`.
7–9. **EWM, SAW, ranking** (terpisah per tugas, hanya dari hasil `val`):

   ```bash
   python -m src.optimization.model_selection --config configs/evaluation/selection.yaml
   ```

   Hasil: `outputs/selection/{detection,depth}_ewm/` dan `outputs/selection/{detection,depth}_saw/`
   (termasuk `ranking.png` dan `selection_report.md`).
10. **Checkpoint terbaik**: `outputs/selection/selected_models.json` berisi checkpoint student KD berperingkat
    tertinggi per tugas; inilah yang dipakai integrasi.

Atau kedua evaluasi sekaligus: `bash scripts/evaluate_all.sh device=0` (split `test` ikut dievaluasi hanya bila
dataset dibuat dengan `split.mode=internal`).

---

## 8. Integrasi dan estimasi jarak

1–2. Memuat checkpoint deteksi dan depth terbaik: otomatis dari `selected_models.json`, atau eksplisit dengan
   `--detection-weights` / `--depth-weights`.
3. Inferensi kedua model pada citra RGB yang sama.
4. Penyesuaian koordinat box ke grid depth (Pers. 3.11–3.12).
5. Inner ROI 60% (margin 20%, minimum 4 px).
6. Filter depth valid (finite, ≠ 0, 0,1–10 m), `R_valid ≥ 0,30`, clip persentil 5–95.
7–8. Median depth = object distance (axial).
9. Evaluasi terhadap ground truth (same-box pada depth GT split `val`, MAE/RMSE/MAPE):

   ```bash
   python -m src.evaluation.evaluate_distance --config configs/evaluation/distance.yaml
   ```

   Untuk jarak hasil pengukuran langsung: isi `data_collection/distance_measurements.csv` lalu tambahkan
   `--set gt_source=measured`. Hasil: `outputs/integration/distance_evaluation/`.

---

## 9. Navigasi

1. ROI navigasi global = 60% tengah frame secara horizontal, seluruh tinggi.
2. Lima sektor sama lebar: FL, L, CTR, R, FR.
3. Overlap box–sektor `Area(B ∩ S) / Area(B) ≥ 0,05`; box di luar ROI diabaikan.
4. Blocked bila jarak < 1,0 m, atau depth invalid (`invalid_depth_blocks`), atau confidence fusion < 0,20
   (`low_confidence_blocks`); ketiganya di `configs/system/inference.yaml`.
5. Prioritas CTR → L → R → FL → FR.
6. CTR = FORWARD, L/FL = TURN_LEFT, R/FR = TURN_RIGHT, semua blocked = STOP.
7. Akurasi navigasi (isi `data_collection/navigation_scenarios.csv` dengan arah sebenarnya tiap skenario):

   ```bash
   python -m src.evaluation.evaluate_navigation --config configs/evaluation/navigation.yaml
   ```

   Hasil: `outputs/integration/navigation_evaluation/` (per skenario, confusion matrix, akurasi).

---

## 10. Testing: gambar, video, webcam, antarmuka

Ketiganya menerima `--detection-weights`, `--depth-weights`, `--device`, `--output`, `--set`.

**Gambar** (berkas atau folder):

```bash
python -m src.inference.predict_image --source path/ke/gambar_atau_folder --output outputs/inference/images
```

**Video**:

```bash
python -m src.inference.predict_video --source path/ke/video.mp4 --output outputs/inference/videos
```

Opsi: `--stride 2`, `--max-frames 300`, `--display`, `--no-save-video`.

**Webcam realtime** (satu kamera RGB, default indeks 0):

```bash
python -m src.inference.webcam --camera 0 --width 1280 --height 720 --output outputs/inference/webcam
```

Tombol jendela: `q` keluar, `s` simpan frame + JSON, `r` mulai/berhenti rekam. `--headless` untuk tanpa jendela.

**Antarmuka** (tab gambar, video, webcam; nama model yang dimuat; tombol "Simpan hasil"):

```bash
python -m app.main --config configs/system/inference.yaml
```

Lalu buka http://127.0.0.1:7860. Hasil yang disimpan: `outputs/inference/interface/`.

---

## 11. Menjalankan seluruh penelitian

| # | Tahap | Perintah | Output |
|---|---|---|---|
| 1 | Environment | §3 | `.venv/`, test lulus |
| 2 | Dataset | `bash scripts/prepare_dataset.sh` | `datasets/sunrgbd_yolo/`, `outputs/dataset/` |
| 3–5 | Training teacher, baseline, KD deteksi | `bash scripts/train_all_detection.sh train.device=0` | `outputs/detection/{DET-01..06,TEACHER-CA}/` |
| 3–5 | Training teacher, baseline, KD depth | `bash scripts/train_all_depth.sh train.device=0` | `outputs/depth/{DEP-01..09,TEACHER-CBAM}/` |
| 6–7 | Evaluasi (val 5.050) | `bash scripts/evaluate_all.sh device=0` | `outputs/evaluation/*/val/` |
| 8–10 | EWM, SAW, pemilihan | `bash scripts/select_models.sh` | `outputs/selection/` |
| 11–15 | Integrasi, jarak, navigasi | `bash scripts/run_integrated_system.sh datasets/sunrgbd_yolo/images/val <satu_citra_benchmark>` | `outputs/inference/images/`, `outputs/integration/` |
| 16–18 | Gambar / video / webcam | §10 | `outputs/inference/{images,videos,webcam}/` |
| 19 | Laporan | `python -m src.utils.report --root outputs --out outputs/reports/experiment_summary/RESEARCH_REPORT.md` | laporan + CSV per ID |

Sebelum memakai data asli, pastikan seluruh tahap berjalan: `bash scripts/smoke_test.sh`.

---

## 12. Troubleshooting

Tabel lengkap: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Ringkas:

| Masalah | Penanganan |
|---|---|
| CUDA out of memory | Turunkan `train.batch` (akumulasi gradien ke 64 otomatis) / `batch` evaluasi; catat perubahan |
| Checkpoint tidak ditemukan | Latih teacher/eksperimen terkait; beri `kd.teacher_weights=...` atau `--detection-weights/--depth-weights` |
| Dataset path tidak valid | Atur `source.*` di `configs/dataset/sunrgbd.yaml` |
| RGB dan depth tidak cocok | Lihat `meta/preparation_failures.csv` dan `outputs/dataset/integrity_issues.csv` |
| Label YOLO invalid | Jalankan ulang `prepare_sunrgbd`, lalu validasi |
| Tensor dimension mismatch | `nc` dan `imgsz` teacher = student; jalankan `pytest tests/test_*kd.py -v` |
| Feature hook tidak bekerja | Kolom `dis_loss`/`kd_feature` 0 → periksa `kd.feature.enabled`, `metadata.json` |
| KD loss NaN | Coba `train.amp=false`; Ultralytics memulihkan dari NaN dengan `last.pt`; semua loss memakai mask valid |
| Depth tidak valid | Periksa kalibrasi `calibration_best.json` (`mode`), kalibrasi ulang `--calibrate-only` |
| Inferensi gagal | Pesan `[modul] penyebab / File / Solusi` dari `InferenceError` |
| Webcam tidak terdeteksi | Cek kabel/izin, `--camera 1`, `ls /dev/video*` |
| Video tidak dapat dibuka | Konversi ke MP4 H.264 dengan ffmpeg |

---

## 13. Pengujian kode dan status verifikasi

```bash
python -m pytest tests/ -v
```

```bash
bash scripts/smoke_test.sh
```

**Hasil di lingkungan pengembangan** (CPU Intel Xeon 2,1 GHz, 4 core, tanpa GPU; Python 3.13.18, torch 2.15.0,
ultralytics 8.4.194, OpenCV 5.0.0):

- `pytest tests/`: **101 test lulus** (96 unit/komponen + 5 integration end-to-end), mencakup: pairing, dekode
  depth, konversi anotasi, validasi box, split mode proposal (5.285/5.050) dan mode internal ber-grup, model loading, forward teacher dan
  student, CA, CBAM, ekstraksi fitur, alignment fitur, loss KD deteksi (feature, prediction) dan depth (output,
  feature, boundary, relative) termasuk aliran gradien hanya ke student, metrik, EWM, SAW, penolakan split test,
  coordinate mapping, inner ROI, filter depth, jarak, overlap sektor, free/blocked, arah, mode head deteksi,
  augmentasi depth, inferensi gambar, inferensi video, penanganan webcam (tidak terdeteksi, frame kosong, kamera
  terputus, resolusi berbeda), checkpoint hilang. Integration test: gate dataset → training teacher/baseline/KD
  (deteksi + depth) → interrupt + resume → kalibrasi pada `val` → evaluasi → EWM-SAW → integrasi → CLI gambar dan
  video dengan model nyata → evaluasi jarak dan navigasi.
- `scripts/smoke_test.sh`: **lulus** (mode split proposal). 17 training (DET-01…06, TEACHER-CA, DEP-01…09,
  TEACHER-CBAM; 1 epoch, imgsz 96, teacher memakai skala n agar cepat), evaluasi `val`, EWM-SAW, CLI gambar dan video, evaluasi jarak
  dan navigasi, benchmark sistem, laporan per ID. Angkanya tidak bermakna ilmiah.

**Belum dapat diverifikasi di lingkungan pengembangan** (perlu dijalankan peneliti):

- Pembacaan berkas SUN RGB-D asli (`allsplit.mat`, `SUNRGBDMeta2DBB_v2.mat`, PNG `depth_bfx`) — formatnya ditiru
  data sintetis; jalankan §4 dan periksa laporannya sebelum training.
- Pengelompokan grup rekaman pada nama folder asli (aturan diuji dengan nama berpola asli, bukan dengan dataset).
- Training penuh 100 epoch, CUDA, AMP, bobot pretrained resmi, pengukuran memori GPU dan FPS GPU.
- Webcam fisik dan antarmuka Gradio di browser (logika kamera diuji dengan kamera tiruan).

---

## 14. Batasan dan keputusan metodologis

Batasan yang dijaga kode: indoor, objek statis, pencahayaan stabil; satu citra RGB monocular sebagai satu-satunya
input inferensi (tanpa LiDAR, stereo, atau sensor RGB-D); SUN RGB-D; satu kelas `objek`; hanya keluarga YOLO26;
evaluasi navigasi hanya melalui antarmuka; tanpa kendali kursi roda fisik, uji keselamatan operasional, deployment
embedded, tracking, SLAM, path planning, hysteresis, atau temporal smoothing.

Ketidaksesuaian antar-sumber dan keputusan yang perlu dikonfirmasi pembimbing ada di
[docs/EXPERIMENT_DESIGN_DECISIONS.md](docs/EXPERIMENT_DESIGN_DECISIONS.md). Yang terpenting untuk revisi proposal:

- **X-1** Tabel 3.1 memuat 3 model per tugas; implementasi memuat 6 (deteksi) dan 9 (depth) alternatif.
- **X-2** Object ROI Distillation dihapus; Full KD = 4 komponen (Pers. 3.9–3.10 perlu direvisi).
- **S-0** Split proposal train 5.285 / val 5.050 untuk semua model. Karena `val` dipakai untuk seleksi
  (early stopping, kalibrasi, EWM-SAW) sekaligus pelaporan, metrik model berpotensi sedikit optimistis; ini perlu
  ditulis sebagai keterbatasan di BAB 3. Uji independen: data indoor sendiri (RM 3–4).
- **C-1** Kalibrasi log-affine di-fit pada split `val`.
- **A-1/A-2** CA dan CBAM pada P3/P4/P5 teacher; DET-05 = Prediction KD, DEP-08 = Output KD.
- **K-1** Rumus dan bobot Prediction KD (tidak ada di proposal).
- **D-6** Mode head YOLO26 (one-to-many + NMS IoU 0,45 sebagai default).
- **H-5** Augmentasi zoom dimatikan untuk depth.
- **M-2, I-1** Koreksi Pers. 3.28 dan 3.17.
- **N-1** Definisi "confidence score hasil fusion".
- **E-3** Definisi kriteria memori GPU.

## Lisensi

AGPL-3.0, mengikuti Ultralytics yang dipakai dan diperluas (lihat `LICENSE` dan
[docs/SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md)).
