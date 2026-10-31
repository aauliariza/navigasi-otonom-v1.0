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
