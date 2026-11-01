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
| `[model_stats] thop.profile gagal` / GFLOPs = 0 | Paket `thop` lama menimpa `ultralytics-thop` (keduanya memasang modul `thop`) | `pip uninstall -y thop && pip install --force-reinstall ultralytics-thop` |
| CUDA out of memory saat training | Batch terlalu besar; student KD memuat teacher x di GPU yang sama | Turunkan `train.batch` (akumulasi gradien ke `nbs = 64` otomatis, lihat TRAINING_GUIDE §5); tutup proses GPU lain (`nvidia-smi`); sebagai alternatif `train.amp=true` |
| CUDA out of memory saat evaluasi | `batch` evaluasi terlalu besar | `--set batch=4` pada `evaluate_detection` |
| `Checkpoint ... not found`, `[model_loader] checkpoint model ... tidak ditemukan` | Eksperimen belum dilatih / path salah | Latih eksperimen terkait; untuk inferensi beri `--detection-weights` / `--depth-weights` atau jalankan `model_selection` |
| `SUN RGB-D root not found`, `Official split not found` | Path dataset salah | Atur `source.sunrgbd_root`, `source.split_file`, `source.bbox_meta` di `configs/dataset/sunrgbd.yaml` atau via `--set` |
| `resolution mismatch rgb=... depth=...`, `RGB without depth` | Berkas RGB/depth tidak cocok atau hilang | Periksa `datasets/sunrgbd_yolo/meta/preparation_failures.csv` dan `outputs/dataset/integrity_issues.csv`; ekstrak ulang `SUNRGBD.zip` |
| `label: coordinates outside [0,1]`, `boxes with side < 4px`, `class id != 0` | Label YOLO tidak valid (diubah manual / konversi lama) | Jalankan ulang `python -m src.data.prepare_sunrgbd` (folder label ditulis ulang), lalu validasi |
| `group_leakage` pada validasi | Grup rekaman muncul di train dan val (mis. `group_overrides` tidak konsisten) | Periksa `meta/splits.json` dan CSV `group_overrides`, jalankan ulang preprocessing |
| `Teacher/student head outputs are not anchor-aligned` | `imgsz` atau `nc` teacher berbeda dari student | Latih teacher dan student dengan `nc = 1` dan `imgsz` yang sama (Tabel 3.7) |
| Tensor dimension mismatch pada Feature KD | Tap layer berubah (arsitektur kustom) | `python -m pytest tests/test_detection_kd.py tests/test_depth_kd.py -v`; tap diturunkan otomatis dari input head (`get_distill_layers`), jangan menulis indeks layer manual |
| Kolom `dis_loss` / `kd_feature` selalu 0 | Hook fitur tidak menangkap output (komponen tidak aktif atau model dibungkus ulang) | Pastikan `kd.feature.enabled: true`; cek `metadata.json` → `kd_config`; uji `models/common/feature_hooks.py::capture_features` pada checkpoint |
| Loss KD `nan` / `inf` | AMP overflow, teacher depth 0 di area invalid, lr terlalu besar | Semua loss memakai mask valid dan `clamp_min`; coba `train.amp=false`, periksa `results.csv` kolom `train/kd_*`, Ultralytics otomatis memulihkan dari NaN dengan memuat ulang `last.pt` |
| Depth map berisi nilai tidak wajar / kalibrasi `scale_only_fallback` | Model belum konvergen | Periksa kurva loss; kalibrasi ulang dengan `--calibrate-only`; nilai di luar 0,1–10 m tidak dipakai untuk jarak |
| `[pipeline] frame tidak valid` | Gambar kosong / grayscale | Gunakan citra BGR 3 kanal |
| `[detection_model] gagal memuat model deteksi` | Checkpoint rusak atau bukan YOLO26 | Pakai `weights/best.pt` hasil training repository ini; `import models` diperlukan untuk teacher ber-attention |
| `[webcam] kamera '0' tidak terdeteksi atau gagal dibuka` | Kamera tidak tersambung / dipakai aplikasi lain / izin | Periksa kabel dan izin; coba `--camera 1`; di Linux cek `ls /dev/video*` |
| `[webcam] ... frame gagal berturut-turut; kamera dianggap terputus` | Kabel lepas / driver | Sambungkan ulang; naikkan `--max-failed-frames` bila kamera lambat |
| Resolusi webcam berbeda dari `--width/--height` | Kamera tidak mendukung resolusi itu | Normal; resolusi aktual dipakai dan dicatat di `webcam_session.json` |
| `[predict_video] video tidak dapat dibuka` | Codec tidak didukung OpenCV | Konversi: `ffmpeg -i in.mov -c:v libx264 out.mp4` |
| `GPU '0' diminta tetapi CUDA tidak tersedia` | Driver/PyTorch CPU-only | Inferensi berjalan di CPU (peringatan); pasang PyTorch CUDA untuk FPS penelitian |
| `contains results of the TEST split` saat model selection | Hasil evaluasi `split=test` (mode internal) diberikan ke EWM-SAW | Evaluasi dengan `split=val` untuk seleksi; test hanya untuk laporan akhir |
| `KeyError: 3` di `feature_kd` (`detection_teacher_student.py`) | Ultralytics lebih lama dari 8.4.194 (mis. 8.4.138) menyimpan fitur KD dengan kunci nomor layer | `pip install "ultralytics==8.4.194"`; repository kini menolak versi lama sejak awal (`src/utils/versions.py`) |
| `[training] Teacher ... dilatih dengan Ultralytics X` | Teacher dilatih dengan versi Ultralytics lain | Latih ulang teacher dengan versi terpasang agar semua eksperimen memakai versi yang sama |
