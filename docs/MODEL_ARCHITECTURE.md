# Arsitektur Model

## YOLO26 (deteksi), `yolo26.yaml`, skala n dan x

- **Backbone:** layer 0–10. Konv stride-2, C3k2, SPPF, dan C2PSA, menghasilkan P3 (stride 8), P4 (16), P5 (32).
- **Neck:** layer 11–22, FPN top-down + PAN bottom-up. Keluaran P3, P4, P5 ada di layer **16, 19, 22**.
- **Head:** layer 23, `Detect` anchor-free, end-to-end (cabang one2many + one2one), DFL-free (`reg_max = 1`),
  `nc = 1`.

## YOLO26-Depth, `yolo26-depth.yaml`

Backbone dan neck sama. Head `Depth` di layer 23 menerima input 16/19/22:

- proyeksi 1×1 ke `c_mid = 256`;
- fusi top-down dengan refine 3×3;
- ConvTranspose ×2, menghasilkan log-depth pada ¼ resolusi input, di-clamp ke [−4, 5];
- `depth = exp(cal_a · log d + cal_b)` saat eval.

## Teacher ber-attention (`models/common/model_utils.py::attention_yaml`)

| Layer | YOLO26x + CA | YOLO26x-Depth + CBAM |
|---|---|---|
| 0–22 | sama dengan YOLO26x | sama dengan YOLO26x-Depth |
| 23 | `CoordAtt(C_P3, 32)` dari 16 | `CBAM(C_P3, 16, 7)` dari 16 |
| 24 | `CoordAtt(C_P4, 32)` dari 19 | `CBAM(C_P4, 16, 7)` dari 19 |
| 25 | `CoordAtt(C_P5, 32)` dari 22 | `CBAM(C_P5, 16, 7)` dari 22 |
| 26 | `Detect` dari [23, 24, 25] | `Depth` dari [23, 24, 25] |

Jumlah kanal diturunkan otomatis dari skala model. YAML hasil generate disimpan di folder eksperimen
(`model_yolo26x_ca.yaml`). Bobot pretrained ditransfer oleh `transfer_weights`: layer 0–22 berdasarkan nama, head
lama 23 dipetakan ke 26 dengan menyisipkan `nn.Identity`. Attention selalu diinisialisasi acak.

## Teacher–student

```
             ┌───────────── teacher (eval, requires_grad=False, no_grad) ──────────────┐
 img ──┬───► │ ... ─ 16/19/22 (atau 23/24/25 attention) ─► head ─► skor/box atau depth  │
       │     └───────────────┬────────────────────────────────┬────────────────────────┘
       │                     │ hook fitur                     │ hook head
       │     ┌─────── projector (training-only) ───────┐      │
       └───► │ student ─ 16/19/22 ─► head ─► prediksi   │──► L_GT + Σ λ_k L_KD^(k)
             └──────────────────────────────────────────┘
```

| | Deteksi (`DetectionKDModel`) | Depth (`DepthKDModel`) |
|---|---|---|
| Tap fitur | input Detect head | input Depth head |
| Proyektor | student → teacher (Ultralytics), 2 × konv 1×1 | teacher → student (proposal), 2 × konv 1×1 |
| Loss KD | Feature (score-weighted L2, `dis`) dan/atau Prediction | Output, Feature, Boundary, Relative |
| Yang disimpan | `best.pt` = YOLO26n saja | `best.pt` = YOLO26n-Depth saja (+ kalibrasi) |

Tidak ada komponen training-only yang ikut ke student saat inferensi. Proyektor dan teacher dibuang oleh
`strip_optimizer`, sehingga parameter dan FLOPs student KD identik dengan baseline.
