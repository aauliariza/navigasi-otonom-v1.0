# Metodologi dan Keputusan Implementasi

Dokumen ini menjelaskan bagaimana BAB 3 proposal diterjemahkan menjadi kode. Setiap ketidaksesuaian atau hal yang
belum ditetapkan proposal dicatat sebagai **keputusan** (kode A/D/K/C/E/M/I/N/H). Keputusan bertanda **⚠ perlu
konfirmasi** berpotensi memengaruhi hasil penelitian. Semuanya dapat diubah lewat YAML tanpa mengubah kode.

## 1. Alur penelitian (waterfall, Gambar 3.1 dan 3.3)

```
SUN RGB-D (split resmi 5.285 / 5.050)
  ├─ prepare_sunrgbd: pairing RGB-depth, dekode depth (m), valid mask, label YOLO 1 kelas
  └─ validate_dataset: integrity gate (training ditolak bila gagal)
Deteksi:  YOLO26x ─┬─ YOLO26n baseline
                   ├─ Feature KD ─ Prediction KD
          YOLO26x+CA ─┬─ KD (Feature) ─ Feature+Prediction KD
Depth:    YOLO26x-Depth ─┬─ YOLO26n-Depth baseline
                         ├─ Output / Feature / Boundary / Relative / Full KD
          YOLO26x-Depth+CBAM ─┬─ KD (Output) ─ Full KD
          (seluruh model depth: kalibrasi log-affine pasca-training)
Evaluasi (val) → benchmark → EWM → SAW → student KD terpilih (deteksi + depth)
Integrasi 1 citra RGB → alignment → inner ROI 60% → jarak (median) → 5 sektor → FORWARD/TURN_LEFT/TURN_RIGHT/STOP
Evaluasi jarak (MAE/RMSE/MAPE) dan akurasi navigasi → interface
```

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

## 3. Daftar keputusan

### Arsitektur dan desain eksperimen

- **A-1 ⚠ perlu konfirmasi: CA dan CBAM tidak ada dalam proposal.** Tabel 3.1 proposal hanya berisi 3 model per
  tugas, tanpa attention. Prompt bagian 5 mewajibkan 6 eksperimen deteksi (termasuk CA) dan 9 eksperimen depth
  (termasuk CBAM), jadi keduanya diimplementasikan. Penempatannya: satu blok attention pada setiap fitur piramida
  P3/P4/P5 (output layer 16, 19, 22) tepat sebelum head, sebagai layer 23–25. Head pindah ke layer 26.
  Alasannya:
  - Fitur inilah yang dikonsumsi head dan yang didistilasi. KD dengan teacher ber-attention membaca fitur hasil
    attention pada level yang sama, sehingga kompatibel dengan tap student 16/19/22.
  - Layer 0–22 tidak berubah indeks, jadi bobot pretrained resmi tetap dapat dimuat.
  - Student tidak pernah dimodifikasi, sehingga tidak ada biaya inferensi tambahan pada student.

  Biaya attention pada teacher (dihitung `model_stats`, 640 px, sebelum *fuse*):

  | Teacher | Parameter | GFLOPs |
  |---|---|---|
  | YOLO26x | 58.810.878 | 209,06 |
  | YOLO26x + CA | 58.939.314 (+0,22%) | 209,19 |
  | YOLO26x-Depth | 55.841.569 | 210,23 |
  | YOLO26x-Depth + CBAM | 56.009.797 (+0,30%) | 210,23 |

  Nilai final akan diukur ulang oleh `evaluate_*` dengan `--include-extra`. Kebutuhan memori GPU teacher
  ber-attention tercatat di `gpu_memory.csv` setiap eksperimen.
- **A-2 Jenis KD pada eksperimen "teacher + attention → student dengan KD".**
  - **5.1-5 (CA → YOLO26n):** memakai mekanisme KD framework yang ditetapkan proposal (Sub-bab 3.2.4.2), yaitu
    Feature KD score-weighted L2 dengan `dis = 6.0`. Satu-satunya perbedaan terhadap 5.1-3 adalah teacher ber-CA,
    sehingga efek CA dapat diisolasi. Perbedaan terhadap 5.1-6 adalah ketiadaan Prediction KD.
  - **5.2-8 (CBAM → YOLO26n-Depth):** memakai Depth Output Distillation saja (KD berbasis respons paling dasar,
    Pers. 2.9). Satu-satunya perbedaan terhadap 5.2-3 adalah teacher ber-CBAM. 5.2-9 memakai keempat komponen.

  Kedua pilihan dapat diganti di `configs/detection/ca_kd.yaml` dan `configs/depth/cbam_kd.yaml`.
- **A-3 Teacher dan student memakai satu kelas (`nc = 1`).** Detect head teacher dan student sama-sama
  menghasilkan skor satu kelas, sehingga Prediction KD dapat dilakukan anchor-per-anchor.

### Dataset

- **D-1 ⚠ perlu konfirmasi: sumber depth `depth_bfx` (refined) atau `depth` (raw).** Proposal hanya menyebut
  depth uint16 dengan Pers. 3.2, yang berlaku untuk keduanya. Default yang dipakai adalah `depth_bfx`, berkas yang
  dirujuk metadata toolbox SUN RGB-D dan konverter Ultralytics. Untuk memakai depth raw sensor, ganti dengan
  `source.depth_dir: depth`.
- **D-2 Nama subset.** `alltrain` (5.285) menjadi `train` dan `alltest` (5.050) menjadi `val`, sesuai Tabel 3.6.
  Tidak ada test set tambahan. Pipeline gagal bila jumlah pasangan valid ≠ 5.285/5.050 (`split.strict_counts`).
  Bila ada data asli yang memang rusak, keputusan untuk mengecualikannya harus diambil peneliti secara eksplisit
  dengan `strict_counts: false` dan dicatat.
- **D-3 Depth di luar 0,001–10 m dibuat invalid (0), bukan di-clip ke 10 m.** Clipping akan menciptakan
  ground truth palsu bernilai tepat 10 m yang kemudian dianggap valid oleh loss dan evaluasi. Proposal menyatakan
  "nilai di luar rentang tersebut tidak digunakan" (Sub-bab 3.2.4.3). Perilaku konverter Ultralytics tersedia
  sebagai opsi `depth.out_of_range: clip`.
- **D-4 Sumber box: `SUNRGBDMeta2DBB_v2.mat` (`gtBb2D = [x, y, w, h]`).** Proposal menyebut 146.617 anotasi 2D.
  Angka itu identik dengan jumlah poligon 2D di paper SUN RGB-D, sehingga jumlah box di berkas Meta2DBB bisa
  berbeda. Jumlah aktual dilaporkan di `datasets/sunrgbd_yolo/meta/preparation_report.json`. Alternatifnya
  adalah poligon `annotation2Dfinal/index.json` (`source.bbox_source: annotation2d`).
- **D-5 Koordinat box tidak digeser −1** (meskipun berasal dari MATLAB yang berbasis 1). Pers. 3.3 dipakai secara
  literal. Selisihnya maksimum 1 piksel.

### Knowledge Distillation

- **K-1 ⚠ perlu konfirmasi: rumus dan bobot Prediction KD.** Proposal tidak memberi rumus Prediction KD. Yang
  diimplementasikan adalah soft-label BCE kelas dan L1 box berbobot confidence teacher, pada kedua cabang head
  YOLO26 (lihat REFERENCES_AND_IMPLEMENTATION §3). Bobot default 1,0/1,0/1,0 dengan T = 1 bersifat eksperimental.
  Karena besarnya loss berbeda dari Feature KD (`dis = 6`), kolom `pkd_cls`/`pkd_box` di `results.csv` perlu
  diperiksa setelah beberapa epoch.
- **K-2 Proyektor feature depth: teacher → student**, sesuai proposal ("penyesuaian channel teacher ke student").
  Proyektor yang dilatih pada sisi teacher dapat mencari solusi trivial, sehingga kedua fitur dinormalisasi L2 per
  lokasi sebelum MSE. Arah FitNets tersedia sebagai `direction: student_to_teacher`. Feature KD deteksi memakai
  arah Ultralytics (student → teacher) karena proposal merujuk mekanisme framework.
- **K-3 Pasangan "sama" pada Relative KD.** Bila Pers. 2.14 diterapkan literal pada `r = 0`, hasilnya konstanta
  `log 2` tanpa gradien. Karena itu dipakai suku `ΔS²` dari Chen et al. (2016). Opsi `equal_pair_loss: ignore`
  mereproduksi bentuk literal. Margin `m` default 0, karena proposal tidak memberi nilainya.
- **K-4 Ruang depth teacher.** Teacher dijalankan dalam mode eval, sehingga head menerapkan kalibrasinya sendiri
  (`teacher_space: calibrated`). Opsi `raw` membatalkan kalibrasi tersebut.
- **K-5 Parameter loss tanpa nilai di proposal:**
  - SmoothL1 β = 0,1 (output dan boundary).
  - Boundary dihitung pada depth linear, sesuai Pers. 2.11.
  - Normalisasi feature: L2.
  - Seed sampler relative: 42.
- **K-6** Teacher harus sudah dilatih. Path `best.pt`-nya diturunkan dari `kd.teacher_config` dan SHA-256-nya
  dicatat di `metadata.json` student.

### Kalibrasi depth

- **C-1 ⚠ perlu konfirmasi: kalibrasi log-affine di-fit pada split train.** Proposal mengevaluasi setelah
  kalibrasi tetapi tidak menyebut data fitting-nya. Fitting pada split validasi lalu mengevaluasi pada split yang
  sama adalah kebocoran data. Ultralytics melakukan hal itu, jadi kalibrasi otomatisnya dimatikan dan diganti.
  Default: maksimum 1.000 citra train dan 20.000 piksel per citra, dengan seed 42.
- **C-2** Bila slope `a ≤ 0` (model degeneratif), dipakai fallback scale-only `a = 1` dan kejadiannya dicatat
  (`mode`).

### Evaluasi

- **E-1 Precision, Recall, F1 utama dihitung pada titik operasi inferensi:** conf 0,25, NMS IoU 0,45, maksimum
  100 deteksi, pencocokan IoU 0,5, dengan TP/FP/FN eksak. mAP50/mAP50-95 dihitung dengan protokol standar
  Ultralytics (conf 0,001). P/R pada F1 maksimum versi Ultralytics juga disimpan.
- **E-2 Metrik depth dihitung pooled atas semua piksel valid** (N = jumlah piksel valid, Pers. 3.34), tanpa median
  alignment per citra. Prediksi di-clamp ke [0,001; 10]. Rata-rata per citra juga dilaporkan.
- **E-3 ⚠ perlu konfirmasi: kriteria "penggunaan memori GPU" untuk EWM-SAW** memakai puncak memori GPU saat
  inferensi (benchmark model-only, B = 1). Memori saat training tercatat di `gpu_memory.csv`. Memori training
  student KD ikut mencakup teacher, sehingga tidak sebanding untuk menilai efisiensi model. Bila evaluasi
  dijalankan di CPU, kriteria memori GPU tidak tersedia dan dikeluarkan dari EWM (`missing_policy`), dengan
  laporan eksplisit.
- **E-4 FPS model-only:** B = 1, FP32, input 640, 50 warm-up dan 300 iterasi, dengan sinkronisasi CUDA. FPS
  pipeline dan end-to-end diukur terpisah (lihat `src/evaluation/benchmark.py`).

### Pemilihan model

- **M-1 EWM pada nilai mentah `x_ij`**, sesuai Pers. 3.29. EWM memberi bobot besar pada kriteria dengan sebaran
  relatif terbesar. Bila metrik akurasi antar-model hampir sama sementara FPS bervariasi, bobot akan didominasi
  FPS. Oleh karena itu:
  - FPS perlu diukur dengan iterasi cukup pada perangkat yang sama.
  - Analisis sensitivitas dapat dijalankan dengan `ewm_basis: normalized`.
- **M-2 Koreksi Pers. 3.28.** Proposal menulis `r = x / min(x)` untuk kriteria cost. Nilai itu ≥ 1 dan *naik*
  bersama biaya, sehingga model terburuk mendapat skor tertinggi. Yang diimplementasikan adalah normalisasi cost
  SAW standar `r = min(x) / x`.
- **M-3** Integrasi memakai **student KD berperingkat tertinggi** pada masing-masing tugas (Sub-bab 3.2.5.3:
  "YOLO26n dan YOLO26n-Depth hasil KD"). Model terbaik keseluruhan (yang bisa saja teacher atau baseline) tetap
  dilaporkan. Opsi `integration_pool: all` mengizinkan model non-KD.

### Integrasi dan navigasi

- **I-1 Koreksi Pers. 3.17.** Proposal menulis `y2' = y1 − 0,2h`. Yang diimplementasikan adalah `y2 − 0,2h`
  (60% bagian tengah, sesuai teks).
- **I-2** Depth map dipakai pada resolusi keluaran model (¼ dari 640 × 640, hasil stretch). Box diselaraskan
  dengan Pers. 3.11–3.12, dan ukuran ROI minimum 4 piksel diterapkan pada grid depth. Opsi
  `depth.resolution: image` meng-upsample depth ke resolusi citra.
- **I-3** Rentang depth valid untuk jarak adalah 0,1–10 m (Sub-bab 3.2.4.6), berbeda dari rentang ground truth
  training/evaluasi 0,001–10 m. Keduanya mengikuti proposal.
- **I-4** Persentil 5–95 di-*clamp* (`clip_mode: clip`) sebelum median. Untuk median, clamp simetris tidak
  mengubah hasil, tetapi langkah ini tetap dipertahankan sesuai proposal. Alternatif `trim` tersedia.
- **N-1 ⚠ perlu konfirmasi: "confidence score hasil fusion"** tidak didefinisikan proposal. Default memakai
  confidence deteksi. Alternatifnya `detection_x_valid_ratio`. Karena deteksi sudah difilter conf ≥ 0,25, aturan
  `< 0,20` dengan definisi default tidak akan pernah aktif.
- **N-2** Urutan sektor kiri→kanan: FL, L, CTR, R, FR, sesuai urutan penyebutan proposal.
- **N-3** Kondisi "jarak < 1,0 m" diterapkan ketat; objek tepat 1,0 m tidak memblokir.
- **N-4 Ground truth navigasi:**
  - (a) `scenarios`: arah sebenarnya dicatat peneliti untuk skenario indoor nyata (Sub-bab 3.2.5.4). Ini sumber
    utama.
  - (b) `sunrgbd`: arah referensi diturunkan dari box GT dan jarak dari depth GT dengan aturan yang sama. Ini
    referensi turunan, bukan label manusia, dan berguna untuk mengisolasi kesalahan model.

### Hyperparameter

- **H-1** Ukuran input depth 640 × 640 (Tabel 3.8), bukan 768 seperti dokumentasi Ultralytics.
- **H-2 ⚠ perlu konfirmasi:** inisialisasi dari bobot pretrained resmi Ultralytics (`yolo26{n,x}.pt` COCO dan
  `yolo26{n,x}-depth.pt`). Proposal tidak menyebutnya. Untuk melatih dari nol, gunakan `model.weights: null`.
  Teacher dan student memakai kebijakan yang sama.
- **H-3** Augmentasi default Ultralytics dipertahankan identik untuk semua eksperimen dalam satu tugas (deteksi:
  mosaic dengan `close_mosaic = 10`; depth: tanpa mosaic, sesuai `DepthDataset`).
- **H-4** `deterministic: true` dan seed 42. Beberapa operasi CUDA tidak memiliki implementasi deterministik
  (PyTorch akan memberi peringatan), sehingga hasil antar-run dapat berbeda sangat kecil.
