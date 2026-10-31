# Keputusan Desain Eksperimen

Dokumen ini mencatat setiap ketidaksesuaian antara **daftar eksperimen pada prompt**, **proposal (BAB 1–3)**,
**kode resmi/paper**, dan **repository referensi**, beserta dampak dan keputusan yang diambil. Hierarki sumber
kebenaran yang dipakai:

1. Daftar eksperimen eksplisit pada prompt (konfigurasi model yang harus dibuat).
2. Proposal tesis (metodologi utama).
3. Repository resmi dan paper ilmiah (implementasi teknis).
4. Repository referensi `autonomous-wheelchair-v1.0` (inspirasi).

Setiap hal yang belum ditetapkan proposal dicatat sebagai **keputusan** (kode S/A/D/K/C/E/M/I/N/H). Keputusan
bertanda **⚠ perlu konfirmasi** berpotensi memengaruhi hasil penelitian; semuanya dapat diubah lewat YAML tanpa
mengubah kode. Pemetaan persamaan proposal ke kode ada di [METHODOLOGY_MAPPING.md](METHODOLOGY_MAPPING.md).

## 0. Ketidaksesuaian utama antar-sumber (ringkasan)

| # | Sumber A | Sumber B | Keputusan (berdasarkan hierarki) | Dampak pada proposal |
|---|---|---|---|---|
| X-1 | Proposal Tabel 3.1: 3 model per tugas | Prompt: DET-01…06 dan DEP-01…09 | Seluruh 6 + 9 eksperimen (plus 2 teacher ber-attention) diimplementasikan (A-1) | BAB 3 perlu diperluas: Tabel 3.1, 3.9–3.20 memuat 6/9 alternatif, bukan 3; `m` pada Pers. 3.30 menjadi 6 dan 9 |
| X-2 | Proposal Sub-bab 3.2.4.4: 5 komponen KD depth termasuk Object ROI (Pers. 3.10, λ_ROI = 0,2) | Prompt 6.1: 4 komponen, ROI dilarang | Object ROI Distillation **tidak** diimplementasikan; Full KD = output + feature + boundary + relative dengan bobot 0,5/0,2/0,1/0,1 dari Pers. 3.10 | Pers. 3.9–3.10 dan paragraf "lima komponen" (Sub-bab 3.2.4.4 dan 3.2.5.2) perlu direvisi menjadi empat komponen |
| X-3 | Proposal Tabel 3.6: 5.050 = "Validation" | Prompt 7.2 dan pembagian resmi SUN RGB-D: 5.050 = test | `alltest` dipakai sebagai **test** (evaluasi akhir); validasi internal dipisah dari `alltrain` per grup scene (S-1, S-2) | Tabel 3.6 perlu tiga baris (train / validasi internal / test) |
| X-4 | Proposal: CA/CBAM tidak disebut | Prompt 5.6 dan 6.9 | CA pada teacher YOLO26x, CBAM pada teacher YOLO26x-Depth, P3/P4/P5 (A-1) | Perlu sub-bab arsitektur teacher ber-attention dan rujukan Hou et al. (2021), Woo et al. (2018) |
| X-5 | Proposal Sub-bab 3.2.4.2: KD deteksi = mekanisme framework (score-weighted L2, `dis = 6`) | Prompt: Feature KD dan Prediction KD terpisah | Feature KD = mekanisme framework (Ultralytics `DistillationModel`); Prediction KD = implementasi baru (K-1) | Sub-bab 3.2.4.2 perlu memuat rumus Prediction KD |
| X-6 | Proposal: kalibrasi log-affine "setelah training", data fitting tidak disebut | Prompt 10: parameter dari data validasi | (a, b) di-fit pada validasi internal (C-1) | Sub-bab 3.2.4.3 perlu menyebut data fitting kalibrasi |
| X-7 | Proposal Pers. 3.17 `y2' = y1 − 0,2h`, Pers. 3.28 `x / min(x)` | Teks proposal / literatur SAW | Dikoreksi (I-1, M-2) | Kedua persamaan perlu diperbaiki |
| X-8 | Proposal: NMS IoU 0,45 | Prompt 9: jangan memaksakan NMS pada jalur NMS-free | Head mode eksplisit di config; default one-to-many + NMS IoU 0,45 (D-6) | Sub-bab 3.2.4.1 perlu menyebut head YOLO26 yang dipakai |
| X-9 | Repository referensi: hysteresis, majority vote, EMERGENCY_STOP, ROI-KD, Optuna | Prompt 21 butir 10–11 | Tidak dipakai | — |

## 1. Alur penelitian (waterfall, Gambar 3.1 dan 3.3)

```
SUN RGB-D: alltrain 5.285 → train + val (validasi internal, per grup scene) ; alltest 5.050 → test
  ├─ prepare_sunrgbd: pairing RGB-depth, dekode depth (m), valid mask, label YOLO 1 kelas, splits.json
  └─ validate_dataset: integrity gate (training ditolak bila gagal; termasuk cek kebocoran grup train/val)
Deteksi:  DET-01 YOLO26x ─┬─ DET-02 YOLO26n baseline
                          ├─ DET-03 Feature KD ─ DET-04 Prediction KD
          TEACHER-CA (YOLO26x+CA) ─┬─ DET-05 Prediction KD ─ DET-06 Feature + Prediction KD
Depth:    DEP-01 YOLO26x-Depth ─┬─ DEP-02 YOLO26n-Depth baseline
                                ├─ DEP-03 Output / DEP-04 Feature / DEP-05 Boundary / DEP-06 Relative / DEP-07 Full KD
          TEACHER-CBAM (YOLO26x-Depth+CBAM) ─┬─ DEP-08 Output KD ─ DEP-09 Full KD
          (seluruh model depth: kalibrasi log-affine pada val setelah training)
Evaluasi val → benchmark → EWM → SAW → student KD terpilih (deteksi + depth) ; evaluasi akhir pada test
Integrasi 1 citra RGB → alignment → inner ROI 60% → jarak (median) → 5 sektor → FORWARD/TURN_LEFT/TURN_RIGHT/STOP
Evaluasi jarak (MAE/RMSE/MAPE) dan akurasi navigasi (test / skenario indoor) → interface gambar/video/webcam
```

## 2. Daftar keputusan

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
  - **DET-05 (YOLO26x + CA → YOLO26n):** memakai **Prediction KD** sebagai KD dasar, sesuai prompt 5.1/5.6
    ("Baseline prediction KD"). Satu-satunya perbedaan terhadap DET-04 adalah teacher ber-CA, sehingga efek CA
    terisolasi. Perbedaan terhadap DET-06 adalah ketiadaan Feature KD. *(Revisi sesi ini: versi sebelumnya
    repository memakai Feature KD untuk eksperimen ini; diganti agar sesuai daftar eksperimen prompt.)*
  - **DEP-08 (YOLO26x-Depth + CBAM → YOLO26n-Depth):** memakai Depth Output Distillation saja (prompt 6.1).
    Satu-satunya perbedaan terhadap DEP-03 adalah teacher ber-CBAM. DEP-09 memakai keempat komponen.

  Kedua pilihan dapat diganti di `configs/detection/ca_kd.yaml` dan `configs/depth/cbam_kd.yaml`.
- **A-4 Experiment ID.** Folder eksperimen memakai ID prompt: `outputs/detection/DET-01` … `DET-06`,
  `outputs/depth/DEP-01` … `DEP-09`. Kedua teacher ber-attention adalah run pendukung yang dipakai bersama oleh
  dua eksperimen, sehingga disimpan sebagai `outputs/detection/TEACHER-CA` dan `outputs/depth/TEACHER-CBAM`
  (dievaluasi sebagai `extra_models`, tidak ikut EWM-SAW karena bukan alternatif pada daftar prompt).
- **A-3 Teacher dan student memakai satu kelas (`nc = 1`).** Detect head teacher dan student sama-sama
  menghasilkan skor satu kelas, sehingga Prediction KD dapat dilakukan anchor-per-anchor.

### Dataset

- **D-1 ⚠ perlu konfirmasi: sumber depth `depth_bfx` (refined) atau `depth` (raw).** Proposal hanya menyebut
  depth uint16 dengan Pers. 3.2, yang berlaku untuk keduanya. Default yang dipakai adalah `depth_bfx`, berkas yang
  dirujuk metadata toolbox SUN RGB-D dan konverter Ultralytics. Untuk memakai depth raw sensor, ganti dengan
  `source.depth_dir: depth`.
- **S-1 Pembagian data (revisi sesi ini).** `alltrain` (5.285) → `train` + `val` (validasi internal);
  `alltest` (5.050) → `test`. Proposal Tabel 3.6 menamai 5.050 sampel itu "Validation"; sesuai prompt 7.2 dan
  pembagian resmi, sampel tersebut adalah **test** dan hanya dipakai untuk evaluasi akhir. `val` dipakai untuk:
  early stopping dan pemilihan `best.pt` (validator Ultralytics membaca kunci `val`), kalibrasi log-affine (C-1),
  dan EWM-SAW (M-4). Pipeline gagal bila `train + val ≠ 5.285` atau `test ≠ 5.050` (`split.strict_counts`). Bila
  ada data asli yang memang rusak, pengecualiannya harus diputuskan eksplisit dengan `strict_counts: false` dan
  dicatat. *Versi sebelumnya repository memakai alltest sebagai `val` sehingga early stopping melihat data test;
  ini telah diperbaiki.*
- **S-2 ⚠ perlu konfirmasi: validasi internal 10% dari alltrain, dipisah per grup rekaman.** Proposal tidak
  menyebut ukuran validasi internal. Default `split.internal_val.fraction = 0.10` dengan seed 42, diambil per
  sensor (komposisi sensor tetap). Grup (`src/data/create_split.py::scene_group_id`): satu lokasi SUN3D
  (`xtion/sun3ddata/<lokasi>`), satu sesi rekaman ber-timestamp (`kv2/kinect2data`, `kv2/align_kv2`,
  `xtion/xtion_align_data`), selain itu satu sampel. Keterbatasan: folder `kv1/NYUdata/NYUxxxx` dan
  `realsense/*` tidak menyimpan identitas ruangan, sehingga beberapa frame NYU dari ruangan yang sama dapat
  terbagi ke train dan val. Pemetaan yang lebih rinci dapat diberikan lewat `group_overrides` (CSV
  `scene_id,group`). Daftar final disimpan di `datasets/sunrgbd_yolo/meta/splits.json` dan SHA-256-nya dicatat di
  `metadata.json` setiap eksperimen, sehingga seluruh run terbukti memakai split yang sama.
- **D-3 Depth di luar 0,001–10 m dibuat invalid (0), bukan di-clip ke 10 m.** Clipping akan menciptakan
  ground truth palsu bernilai tepat 10 m yang kemudian dianggap valid oleh loss dan evaluasi. Proposal menyatakan
  "nilai di luar rentang tersebut tidak digunakan" (Sub-bab 3.2.4.3). Perilaku konverter Ultralytics tersedia
  sebagai opsi `depth.out_of_range: clip`.
- **D-4 Sumber box: `SUNRGBDMeta2DBB_v2.mat` (`gtBb2D = [x, y, w, h]`).** Proposal menyebut 146.617 anotasi 2D.
  Angka itu identik dengan jumlah poligon 2D di paper SUN RGB-D, sehingga jumlah box di berkas Meta2DBB bisa
  berbeda. Jumlah aktual dilaporkan di `datasets/sunrgbd_yolo/meta/preparation_report.json`. Alternatifnya
  adalah poligon `annotation2Dfinal/index.json` (`source.bbox_source: annotation2d`).
- **D-6 Mode Detect head YOLO26 dan NMS.** Detect head YOLO26 memiliki cabang one-to-many dan one-to-one
  (NMS-free). Pada Ultralytics 8.4.194, checkpoint `.pt` melakukan inferensi dengan cabang **one-to-many + NMS**
  kecuali `model.end2end = True` diset eksplisit (diperiksa pada `nn/modules/head.py`, `utils/nms.py`; diuji di
  `tests/test_detection.py::test_detect_head_mode_selects_nms_path`). Pada jalur NMS-free, IoU threshold tidak
  dipakai sama sekali (hanya conf dan max_det). Agar NMS IoU 0,45 proposal bermakna, default
  `detection.end2end: false` (inferensi) dan `operating_point.end2end: false` (evaluasi), keduanya dicatat di hasil
  (`detection_head`). `end2end: true` tersedia untuk analisis NMS-free; dalam mode itu IoU 0,45 tidak berlaku.
- **D-5 Koordinat box tidak digeser −1** (meskipun berasal dari MATLAB yang berbasis 1). Pers. 3.3 dipakai secara
  literal. Selisihnya maksimum 1 piksel.

### Knowledge Distillation

- **K-1 ⚠ perlu konfirmasi: rumus dan bobot Prediction KD.** Proposal tidak memberi rumus Prediction KD. Yang
  diimplementasikan adalah soft-label BCE kelas dan L1 box berbobot confidence teacher, pada kedua cabang head
  YOLO26 (lihat SOURCE_PROVENANCE.md §3). Bobot default 1,0/1,0/1,0 dengan T = 1 bersifat eksperimental.
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

- **C-1 Kalibrasi log-affine di-fit pada validasi internal (revisi sesi ini).** Prompt 10 mensyaratkan parameter
  kalibrasi dari data validasi, bukan test. (a, b) di-fit dengan kuadrat terkecil bentuk tertutup pada piksel
  valid split `val` (maks. 1.000 citra, 20.000 piksel per citra, seed 42), ditulis ke `cal_a`/`cal_b` checkpoint,
  dan dicatat di `calibration_{best,last}.json`. `calibration.split: test` ditolak dengan error. Kalibrasi
  otomatis Ultralytics (hanya skala) diganti karena proposal Pers. 2.4 memakai dua parameter (a dan b). Hasil test
  dengan demikian tidak memuat scale alignment tersembunyi. *Versi sebelumnya memakai split train.*
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
- **E-5 FLOPs.** Dihitung langsung dengan `thop.profile` (paket `ultralytics-thop`) pada input penuh
  `imgsz × imgsz`, FLOPs = 2 × MAC. `ultralytics.utils.torch_utils.get_flops` tidak dipakai karena mengembalikan 0
  untuk setiap error profiling; nilai 0 akan masuk EWM-SAW sebagai "biaya nol". Kegagalan kini menjadi error
  eksplisit (`models/common/model_utils.py::model_stats`). Nilai pada 640 px terverifikasi sama dengan
  `get_flops` dan dengan tabel resmi (YOLO26n 5,3 GFLOPs pada `nc=1`; YOLO26n-depth 47,2 GFLOPs pada 768 px vs
  46,9 pada dokumentasi).
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

- **M-4 Seleksi hanya pada split validasi.** Hasil evaluasi menyimpan kolom `split`. `model_selection` menolak
  hasil split `test` kecuali `allow_test_split: true` (hanya untuk analisis yang didokumentasikan). Student KD
  untuk integrasi ditentukan dari experiment ID (DET-03…06, DEP-03…09), bukan dari nama tampilan.
- **M-5 Sensitivitas EWM terhadap derau FPS.** Smoke test menunjukkan bahwa bila kolom akurasi hampir identik,
  bobot EWM hampir seluruhnya jatuh pada FPS (kolom dengan variasi terbesar, termasuk variasi pengukuran). Untuk
  data penelitian: ukur FPS pada perangkat yang sama tanpa beban lain, dengan 300 iterasi, dan laporkan analisis
  sensitivitas `ewm_basis: normalized` bila bobot terkonsentrasi pada satu kriteria.

### Integrasi dan navigasi

- **I-1 Koreksi Pers. 3.17.** Proposal menulis `y2' = y1 − 0,2h`. Yang diimplementasikan adalah `y2 − 0,2h`
  (60% bagian tengah, sesuai teks).
- **I-5 Evaluasi jarak dan navigasi pada official test.** `configs/evaluation/{distance,navigation}.yaml`
  memakai `split: test` karena keduanya adalah evaluasi akhir sistem terintegrasi dengan model yang sudah dipilih.
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
- **H-3** Augmentasi ditulis eksplisit di `_common.yaml` dan identik untuk semua eksperimen dalam satu tugas.
  Deteksi: default Ultralytics (HSV, flip horizontal, translate 0,1, scale 0,5, mosaic dengan `close_mosaic = 10`).
  Validasi dan test tidak memakai augmentasi (validator Ultralytics).
- **H-5 ⚠ perlu konfirmasi: augmentasi geometris depth dibatasi (revisi sesi ini).** `RandomPerspective`
  Ultralytics me-warp depth secara spasial tetapi mempertahankan nilai meter (`augment.py::apply_depth`). Zoom
  (`scale`) memperbesar objek, yang secara geometri kamera setara dengan objek lebih dekat, tanpa mengubah label
  depth; ini mengubah skala kedalaman secara tidak sah (prompt 7.5). Untuk depth: `scale = degrees = shear =
  perspective = 0`; yang dipakai: HSV, flip horizontal, translate 0,1 (piksel kosong = 0 = invalid). Konsekuensi:
  hasil depth tidak sebanding langsung dengan training resmi Ultralytics yang memakai `scale = 0,5`.
- **H-6 Batch 8 vs 16 dan akumulasi gradien.** Ultralytics mengakumulasi gradien sampai `nbs = 64` sampel
  (`accumulate = round(64 / batch)`) dan menskalakan weight decay. Batch 8 (teacher) dan 16 (student) pada Tabel
  3.7/3.8 karenanya menghasilkan ukuran batch efektif yang sama (64); yang berbeda hanya statistik BatchNorm per
  langkah dan memori. Ini perilaku bawaan, tidak diubah.
- **H-4** `deterministic: true` dan seed 42. Beberapa operasi CUDA tidak memiliki implementasi deterministik
  (PyTorch akan memberi peringatan), sehingga hasil antar-run dapat berbeda sangat kecil.
