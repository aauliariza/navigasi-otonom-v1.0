# Persiapan Dataset SUN RGB-D

## 1. Berkas yang dibutuhkan (https://rgbd.cs.princeton.edu/)

| Berkas | Isi | Lokasi setelah diekstrak |
|---|---|---|
| `SUNRGBD.zip` (~6,4 GB) | RGB, depth, depth_bfx, intrinsics, anotasi | `datasets/raw/SUNRGBD/{kv1,kv2,realsense,xtion}/...` |
| `SUNRGBDtoolbox.zip` | split resmi | `datasets/raw/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat` |
| `SUNRGBDMeta2DBB_v2.mat` | 2D bounding box | `datasets/raw/SUNRGBDMeta2DBB_v2.mat` |

Unduhan otomatis: `python -m src.data.download_dataset --out datasets/raw`. Bila server tidak dapat dijangkau,
unduh ketiga berkas lewat browser dan tempatkan sesuai kolom terakhir tabel di atas.

## 2. Preprocessing (`python -m src.data.prepare_sunrgbd`)

1. **Split resmi + validasi internal.** `allsplit.mat`: `alltrain` (5.285) dan `alltest` (5.050). Path
   dinormalisasi menjadi identitas scene (`kv1/NYUdata/NYU0001`); scene yang muncul di kedua split menyebabkan
   error. `alltest` → **`test`** (hanya evaluasi akhir). Dari `alltrain`, ±10% diambil sebagai **`val`**
   (validasi internal) per **grup rekaman** (`src/data/create_split.py`): seluruh frame satu lokasi SUN3D, atau
   satu sesi rekaman ber-timestamp, selalu berada di satu sisi; sisanya → **`train`**. Pengambilan per sensor dan
   ber-seed 42. Hasilnya disimpan di `meta/splits.json` (daftar scene, grup, ukuran) dan dipakai semua eksperimen.
   Catatan penamaan: proposal Tabel 3.6 menyebut 5.050 sampel itu "Validation"; di repository ini namanya `test`.
2. **Pairing.** Untuk setiap scene, berkas tunggal `image/*.jpg` dipasangkan dengan `depth_bfx/*.png` dari folder
   scene yang sama. Pemeriksaan yang dilakukan: berkas ada dan unik, dapat didekode, resolusi dan orientasi
   (H × W) identik, serta rasio piksel depth valid > 0.
3. **Dekode depth.** `D_mm = (D_raw >> 3) | (D_raw << 13)` dalam aritmetika uint16 (Pers. 3.2), lalu
   `D_m = D_mm / 1000` (Pers. 3.1).
4. **Valid mask.** Piksel valid bila finite dan 0,001 ≤ D ≤ 10 m. Piksel invalid disimpan sebagai 0.
5. **Penyimpanan.** `depth/<split>/<id>.png` berformat uint16 dengan 1 unit = 1 mm (lossless pada presisi
   sensor). Depth tidak pernah dinormalisasi untuk visualisasi.
6. **Anotasi YOLO.** `gtBb2D [x, y, w, h]` di-clip ke citra. Box di luar citra, tidak valid, atau bersisi < 4 px
   dibuang. Konversi memakai Pers. 3.3–3.8, dengan class ID 0 (`objek`).
7. **Keluaran:**
   - Laporan: `meta/manifest.csv` (split, grup rekaman, sensor, ukuran, jumlah box, rasio depth valid,
     intrinsics fx, fy, cx, cy per citra), `meta/splits.json`, `meta/preparation_report.json`,
     `meta/preparation_failures.csv`.
   - YAML dataset: `sunrgbd_detect.yaml` (nc = 1, names {0: objek}) dan `sunrgbd_depth.yaml` (depth_scale 1000,
     min/max depth).

Struktur hasil:

```
datasets/sunrgbd_yolo/
├── images/{train,val,test}/<id>.jpg   # dipakai bersama oleh deteksi dan depth
├── labels/{train,val,test}/<id>.txt   # YOLO, kelas 0
├── depth/{train,val,test}/<id>.png    # uint16 mm, 0 = invalid
├── meta/manifest.csv, splits.json, preparation_report.json, preparation_failures.csv
├── sunrgbd_detect.yaml
└── sunrgbd_depth.yaml
```

Satu citra RGB dipakai oleh deteksi (`labels/`) dan depth (`depth/`) dengan identitas yang sama. Ultralytics
memetakan `images/` ke `labels/` dan `images/` ke `depth/` berdasarkan nama berkas.

## 3. Validasi (`python scripts/data/validate_dataset.py`)

Laporan di `outputs/dataset/`:

- `integrity_report.json`, dengan field `passed`.
- `integrity_issues.csv`.
- `objects_per_image.csv` dan `.png`.
- `samples/*.jpg` (RGB + box | colormap depth).

Isi laporan: jumlah RGB, depth, dan label per split; pasangan valid; jumlah box; distribusi objek per citra; data
invalid; kesalahan pairing; kebocoran sampel antar train/val/test; kebocoran **grup rekaman** train/val
(`group_leakage_train_val`); dan kecocokan jumlah dengan pembagian resmi (`train + val = 5.285`, `test = 5.050`). **Semua script training menolak berjalan** bila
`integrity_report.json` tidak ada atau `passed: false`.

## 4. Augmentasi

Augmentasi dijalankan oleh dataloader Ultralytics hanya pada `train`; `val` dan `test` tidak diaugmentasi.
Parameter ada di bagian `train:` `configs/{detection,depth}/_common.yaml`. Untuk depth, transformasi geometris
diterapkan sinkron pada RGB dan depth (`DepthDataset`, nearest-neighbour pada depth, piksel kosong = 0 = invalid),
dan zoom/rotasi/shear/perspektif dimatikan agar skala metrik tidak berubah (keputusan H-5).

## 5. Data sintetis (hanya pengujian kode)

`scripts/data/make_synthetic_sunrgbd.py` membuat struktur SUN RGB-D tiruan (depth bit-rotated, `allsplit.mat`,
`SUNRGBDMeta2DBB_v2.mat`, termasuk scene berstruktur SUN3D untuk menguji pengelompokan grup). Data ini hanya untuk
smoke test; hasil apa pun darinya **bukan** hasil penelitian.
