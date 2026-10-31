# Panduan Evaluasi, Pemilihan Model, Integrasi, dan Navigasi

## 1. Evaluasi model (validasi internal → seleksi; test → laporan akhir)

```bash
python -m src.evaluation.evaluate_detection --config configs/evaluation/detection.yaml --include-extra --set split=val device=0
```

```bash
python -m src.evaluation.evaluate_depth --config configs/evaluation/depth.yaml --include-extra --set split=val device=0
```

Ulangi dengan `split=test` untuk evaluasi akhir (atau `bash scripts/evaluate_all.sh device=0` untuk keduanya).
Hasil: `outputs/evaluation/<task>/<split>/`, salinan `metrics.json` di `outputs/<task>/<ID>/evaluation/<split>/`.

| Deteksi | Cara hitung |
|---|---|
| mAP50, mAP50-95 | Validator Ultralytics (conf 0,001), satu kelas → mAP = AP |
| Precision, Recall, F1 | Titik operasi inferensi: conf 0,25, NMS IoU 0,45 (head one-to-many), max 100, pencocokan greedy IoU ≥ 0,5; TP/FP/FN eksak disimpan |
| FPS | Model-only, B = 1, FP32, 640 px, 50 warm-up + 300 iterasi, sinkronisasi CUDA |
| Parameter, FLOPs | Model inferensi ter-*fuse*; FLOPs = 2 × MAC (`thop`) pada 640 px |
| Memori GPU | `torch.cuda.max_memory_allocated` selama benchmark (kosong pada CPU → dikeluarkan dari EWM dan dilaporkan) |

| Depth | Cara hitung |
|---|---|
| δ1, δ2, δ3, AbsRel, SqRel, RMSE, RMSE log | Pers. 3.34–3.38, pooled atas piksel valid GT 0,001–10 m, pada resolusi GT, setelah kalibrasi log-affine yang di-fit pada `val` |
| Visual | `samples/*.jpg` = RGB, GT, prediksi, peta error AbsRel; `depth_maps/*_{pred,gt}.npy` |

## 2. EWM + SAW

```bash
python -m src.optimization.model_selection --config configs/evaluation/selection.yaml
```

Membaca `outputs/evaluation/<task>/val/*_results.csv` (hasil `test` ditolak). Keluaran per tugas:
`outputs/selection/<task>_ewm/` (matriks keputusan, proporsi, entropy, diversifikasi, bobot) dan
`outputs/selection/<task>_saw/` (matriks ternormalisasi, skor, ranking, `ranking.png`, laporan). Checkpoint student
KD berperingkat tertinggi per tugas ditulis ke `outputs/selection/selected_models.json` (input integrasi).

## 3. Integrasi dan estimasi jarak

Pipeline (`src/integration/detector_depth_fusion.py::PerceptionPipeline`):

1. Memuat checkpoint deteksi dan depth terpilih (`selected_models.json`, atau `--detection-weights` /
   `--depth-weights`).
2. Satu citra RGB → YOLO26n (box + confidence) dan YOLO26n-Depth (depth metrik ¼ resolusi 640 × 640).
3. Box dipetakan ke grid depth (Pers. 3.11–3.12).
4. Inner ROI 60% (margin 20%, minimum 4 px).
5. Filter depth valid (finite, ≠ 0, 0,1–10 m), `R_valid ≥ 0,30`.
6. Clip persentil 5–95, median → jarak aksial (mode utama).

Evaluasi akurasi jarak (MAE, RMSE, MAPE):

```bash
python -m src.evaluation.evaluate_distance --config configs/evaluation/distance.yaml
```

- `gt_source: sunrgbd` (default, split `test`): box deteksi yang sama direduksi dengan prosedur identik pada depth
  ground truth (protokol *same-box*).
- `gt_source: measured`: jarak hasil pengukuran langsung (`data_collection/distance_measurements.csv`, protokol di
  `docs/EXPERIMENT_MATRIX.md` §5).

Keluaran `outputs/integration/distance_evaluation/`: `distance_per_object.csv` (box, jarak estimasi, jarak
sebenarnya, error absolut, error %, R_valid), `distance_summary.{json,csv}`, `distance_evaluation.png`.

## 4. Navigasi

1. ROI global = 60% tengah lebar frame, seluruh tinggi (0,2 W ≤ x < 0,8 W).
2. Lima sektor sama lebar 0,12 W: FL, L, CTR, R, FR.
3. Objek menempati sektor bila `Area(B ∩ S) / Area(B) ≥ 0,05`; objek di luar ROI diabaikan.
4. Sektor *blocked* bila ada objek dengan jarak < 1,0 m, atau jarak tidak valid (`invalid_depth_blocks`), atau
   confidence fusion < 0,20 (`low_confidence_blocks`). Ketiganya diatur di `configs/system/inference.yaml`.
5. Prioritas CTR → L → R → FL → FR; sektor bebas pertama menentukan arah (CTR = FORWARD, L/FL = TURN_LEFT,
   R/FR = TURN_RIGHT); semua blocked = STOP.

```bash
python -m src.evaluation.evaluate_navigation --config configs/evaluation/navigation.yaml
```

`gt_source: scenarios` (utama): `data_collection/navigation_scenarios.csv` berisi arah sebenarnya yang ditentukan
peneliti untuk setiap skenario indoor. `gt_source: sunrgbd`: arah referensi diturunkan dari box dan depth ground
truth. Keluaran `outputs/integration/navigation_evaluation/`: per skenario (status lima sektor, arah sistem, arah
sebenarnya, benar/salah), confusion matrix (CSV + PNG), akurasi (Pers. 3.45).

## 5. Laporan gabungan

```bash
python -m src.utils.report --root outputs --out outputs/reports/experiment_summary/RESEARCH_REPORT.md
```

Tabel per experiment ID (val dan test terpisah); metrik yang belum diukur ditulis `NOT MEASURED`.
