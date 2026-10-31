# Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `Dataset integrity report not found` / `check FAILED` | Validasi belum dijalankan atau ada masalah data | Jalankan `python scripts/data/validate_dataset.py`, lalu periksa `outputs/dataset/integrity_issues.csv` |
| Validasi gagal karena jumlah ≠ 5.285/5.050 | Ada scene rusak atau hilang | Lihat `datasets/sunrgbd_yolo/meta/preparation_failures.csv`. Perbaiki atau unduh ulang. Gunakan `--set split.strict_counts=false` hanya bila pengecualian disengaja dan dicatat |
| `Teacher checkpoint not found` | Student KD dijalankan sebelum teacher | Latih teacher terlebih dahulu (urutan di `scripts/train_all_*.sh`) atau beri `--set kd.teacher_weights=path/best.pt` |
| `SUN RGB-D depth must be uint16` | Depth dibaca sebagai citra 8-bit atau berkas salah | Pastikan berkas berasal dari `depth_bfx/` asli |
| Unduhan `yolo26x.pt` gagal | Tidak ada akses ke rilis GitHub Ultralytics | Unduh manual dan isi `model.weights` dengan path lokal, atau gunakan `model.weights=null` |
| CUDA out of memory | Batch terlalu besar (teacher x dan KD memuat dua model) | Turunkan `train.batch` untuk **semua** eksperimen yang dibandingkan, lalu catat perubahannya |
| Peringatan `deterministic algorithm` | Operasi CUDA tanpa versi deterministik | Normal; hasil dapat berbeda sangat kecil antar-run |
| Training terhenti | Listrik atau proses mati | Jalankan perintah yang sama dengan `--resume` (memakai `weights/last.pt`) |
| `Unknown train keys` | Kunci YAML di `train:` bukan argumen Ultralytics | Periksa ejaan atau tambahkan kunci ke `ULTRALYTICS_TRAIN_KEYS` |
| Kolom `gpu_memory_gb` kosong | Evaluasi dijalankan di CPU | Jalankan evaluasi di GPU. Selama kolom kosong, kriteria memori dikeluarkan dari EWM dan dilaporkan |
| `Can't get attribute 'CoordAtt'` saat memuat checkpoint | Modul attention tidak ter-import | Jalankan dari root repository dan `import models` sebelum memuat (`YOLO(...)`) |
| Kalibrasi `mode = scale_only_fallback` | Slope log-affine ≤ 0 (model belum konvergen) | Periksa training, lalu kalibrasi ulang dengan `python -m src.training.train_depth --config <cfg> --calibrate-only` |
| Gradio tidak dapat membuka webcam | Izin browser atau kamera dipakai aplikasi lain | Beri izin kamera, atau gunakan `--mode opencv --camera 0` |
| `ImportError: thop`, GFLOPs = 0 | Paket `thop` lama menimpa `ultralytics-thop` | `pip uninstall thop && pip install -U ultralytics-thop` |
