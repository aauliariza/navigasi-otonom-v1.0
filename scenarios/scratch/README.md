# Skenario training dari nol (scratch)

Folder ini menjalankan **ke-24 eksperimen** (DET-01…10, DEP-01…14) tanpa bobot pretrained. Bobot awal acak; arsitektur
disusun dari `.yaml`, termasuk CA/CBAM.

| | Skenario pretrained | Skenario scratch |
|---|---|---|
| Config | `configs/` | `scenarios/scratch/configs/` |
| Hasil | `outputs/` | `scenarios/scratch/outputs/` |
| Bobot awal | `yolo26{n,x}.pt`, `yolo26{n,x}-depth.pt` | acak (`model.weights: null`) |
| Epoch / patience | 100 / 25 | 300 / 50 |
| lr0 depth (n / x) | 0,0003 / 0,0001 (fine-tuning, keputusan H-8) | 0,001 / 0,0005 (Tabel 3.8) |
| Teacher KD | `outputs/.../best.pt` | `scenarios/scratch/outputs/.../best.pt` |

Selain itu semuanya sama: dataset dan split, hyperparameter lain (termasuk `warmup_bias_lr` 0, H-9), resize letterbox (H-7), augmentasi, bobot KD, serta kode training, evaluasi, dan
EWM-SAW. Tiap config di sini mewarisi config aslinya lewat `_base_`, jadi perubahan pada config asli ikut berlaku di sini.

## Menjalankan

Aktifkan environment yang sama, jalankan dari root repository, sebaiknya di dalam `tmux`:

```bash
bash scenarios/scratch/train_all_detection.sh   # DET-01 ... DET-10
bash scenarios/scratch/train_all_depth.sh       # DEP-01 ... DEP-14
bash scenarios/scratch/evaluate_all.sh device=0 # evaluasi + EWM-SAW
```

- Skrip menolak mulai bila ada training lain yang sedang berjalan, karena GPU hanya satu.
- Skrip boleh dijalankan ulang: run yang selesai dilewati, dan run yang terhenti dilanjutkan dari `last.pt`.
- Override diteruskan ke semua run, mis. `bash scenarios/scratch/train_all_detection.sh train.epochs=100`.
- Satu run saja:

  ```bash
  python -m src.training.train_detection --config scenarios/scratch/configs/detection/baseline.yaml
  ```

Run KD membutuhkan teacher skenario ini. Bila belum ada, training berhenti dengan pesan `Teacher checkpoint not found`,
bukan memakai teacher pretrained.

## Membandingkan hasil

Bandingkan eksperimen dengan ID yang sama:

- `outputs/evaluation/{detection,depth}/val/`
- `scenarios/scratch/outputs/evaluation/{detection,depth}/val/`

Jumlah epoch kedua skenario berbeda dengan sengaja, dan ini perlu dituliskan saat melaporkan perbandingan.

## Pengujian

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest scenarios/scratch/tests
```
