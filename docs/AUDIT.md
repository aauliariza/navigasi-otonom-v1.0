# Audit Repository (Phase 1)

Tanggal audit: 31 Oktober 2026.

## Kondisi awal repository utama `aauliariza/navigasi-otonom-v1.0`

- Branch kerja `claude/gallant-faraday-lx80du` **belum memiliki commit** dan direktori kerja kosong (tanpa berkas
  kode, konfigurasi, maupun dokumentasi).
- Tidak ada kode untuk dipertahankan, diperbaiki, atau dihapus. Seluruh implementasi dibuat baru.

## Repository referensi `aauliariza/autonomous-wheelchair-v1.0`

Isinya sekitar 12.200 baris Python (AGPL-3.0): KD depth (output, feature, boundary, relative, ROI), navigasi lima
sektor dengan hysteresis dan safety layer, Optuna, serta konversi SUN RGB-D. Penilaian terhadap proposal:

| Bagian | Relevan? | Keputusan |
|---|---|---|
| Split resmi `allsplit.mat`, dekode depth | Ya | Konsepnya diadopsi; ditulis ulang dengan gate jumlah 5.285/5.050 dan invalidasi > 10 m |
| Konversi `SUNRGBDMeta2DBB_v2.mat` | Ya | Dokumentasi formatnya dipakai sebagai rujukan |
| Loss KD depth (output/boundary/relative) | Ya | Ditulis ulang mengikuti Pers. 2.9–2.14 proposal |
| ROI-KD | Tidak | Dikecualikan atas instruksi |
| Feature KD lewat `ProjectionBank` di luar model | Bermasalah | Proyektor tidak tercakup optimizer Ultralytics, sehingga diganti subclass `DistillationModel` yang proyektornya ikut dioptimasi dan disimpan |
| Majority-vote hysteresis, `EMERGENCY_STOP`, safety layer | Tidak (di luar proposal) | Tidak dipakai |
| Optuna tuning | Tidak (di luar proposal) | Tidak dipakai |
| Euclidean vs axial distance | Ya (alternatif) | Diimplementasikan sebagai opsi; axial tetap mode utama |

## Ultralytics 8.4.194 (diperiksa langsung dari paket terpasang)

- `ultralytics/nn/distill_model.py`: KD bawaan, berupa Feature KD score-weighted L2 pada input Detect head dengan
  bobot `dis` (default 6,0). Inilah mekanisme yang dirujuk Sub-bab 3.2.4.2 proposal.
- Task `depth` (`yolo26*-depth.yaml`): Depth head pada layer 16/19/22, output log-depth ¼ resolusi, buffer
  kalibrasi `cal_a`/`cal_b`, loss `DepthLoss26`.
- `depth-sunrgbd.yaml` milik Ultralytics memakai **split acak 9.245/1.090**, tidak sesuai proposal. Karena itu
  preprocessing ditulis sendiri.
- Kalibrasi otomatis setelah training Ultralytics di-fit pada split validasi (scale-only). Kalibrasi ini diganti
  (keputusan C-1).
- `DepthMetrics` Ultralytics tidak menghitung Sq_rel dan RMSE log, dan secara default memakai median alignment.
  Metrik sendiri ditulis sesuai Pers. 3.34–3.38.

## Kendala lingkungan pengembangan

- `rgbd.cs.princeton.edu`, rilis GitHub (bobot pretrained), `doi.org`, dan Crossref diblokir proxy. Dataset asli
  dan bobot pretrained tidak dapat diunduh.
- Tidak ada GPU. Semua pengujian dijalankan di CPU dengan data sintetis berformat SUN RGB-D.

---

# Audit sesi kedua (31 Oktober 2026)

Repository utama sudah berisi implementasi dari sesi pertama (commit `4434a42`). Seluruh kode dibaca dan diuji
ulang terhadap prompt yang diperbarui. Kode yang benar dipertahankan; yang diubah hanya bagian berikut.

| Temuan | Dampak | Perbaikan |
|---|---|---|
| `alltest` (5.050) dipakai sebagai `val` Ultralytics | Early stopping dan pemilihan `best.pt` melihat data test (kebocoran) | Split tiga arah train / val (validasi internal per grup scene) / test (`src/data/create_split.py`), keputusan S-1, S-2 |
| Kalibrasi log-affine di-fit pada split train | Prompt 10 meminta data validasi | Di-fit pada `val`; `test` ditolak (C-1) |
| DET-05 memakai Feature KD | Prompt 5.1: DET-05 = "Baseline prediction KD" | DET-05 = Prediction KD dengan teacher CA (A-2) |
| Experiment ID `det_0x_*`/`dep_0x_*` | Prompt 15: folder per ID | `DET-01…06`, `DEP-01…09`, `TEACHER-CA`, `TEACHER-CBAM` (A-4) |
| GFLOPs = 0 bila paket `thop` lama terpasang (`get_flops` menelan error) | Kriteria cost FLOPs bernilai 0 di EWM-SAW tanpa peringatan | Profiling langsung dengan error eksplisit (E-5); terdeteksi oleh smoke test sesi ini |
| Augmentasi zoom (`scale = 0,5`) aktif untuk depth | Mengubah skala kedalaman secara tidak sah (prompt 7.5) | Zoom/rotasi/shear/perspektif = 0 untuk depth; augmentasi eksplisit di YAML (H-5) |
| Mode head YOLO26 (NMS vs NMS-free) implisit | NMS IoU 0,45 bisa tidak bermakna | Mode eksplisit di config, dicatat di hasil (D-6) |
| Tidak ada CLI video dan webcam terpisah, error handling kamera minim | Prompt 14 | `src/inference/{predict_image,predict_video,webcam}.py`, `InferenceError` (modul, penyebab, berkas, solusi) |
| Tidak ada visualisasi ranking, error map depth, grafik jarak/navigasi | Prompt 10, 11.3, 12.6, 13.6 | `ranking.png`, panel error AbsRel, `.npy` prediksi/GT, `distance_evaluation.png`, `confusion_matrix.png` |
| Seleksi dapat memakai hasil test | Prompt 11.3 | Kolom `split` + penolakan hasil test (M-4) |
| Laporan tidak per experiment ID | Prompt 15 (`NOT MEASURED`) | `src/utils/report.py` |

Kendala lingkungan sesi ini: tidak ada GPU, kamera, dataset SUN RGB-D, maupun bobot pretrained resmi;
`download.pytorch.org` diblokir proxy (PyTorch dipasang dari PyPI: torch 2.15.0, Python 3.13).
