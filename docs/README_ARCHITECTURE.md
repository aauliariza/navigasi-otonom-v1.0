# Arsitektur Sistem dan Model

## 0. Arsitektur repository

Alur modul (setiap tahap = modul terpisah; keluaran satu tahap menjadi masukan tahap berikutnya):

```
src/data (prepare_sunrgbd, create_split, validate_dataset)
   │  datasets/sunrgbd_yolo/{images,labels,depth}/{train,val,test} + meta/splits.json
   ├──► src/training/train_detection(_kd)  ── models/detection/*, src/losses/{feature_loss,detection_kd}
   │        outputs/detection/<ID>/weights/best.pt
   └──► src/training/train_depth(_kd)      ── models/depth/*, src/losses/{depth_output,feature,boundary,relative,combined}
            outputs/depth/<ID>/weights/best.pt (+ kalibrasi log-affine)
src/evaluation/evaluate_{detection,depth} + benchmark ──► outputs/evaluation/<task>/{val,test}
src/optimization (EWM, SAW, model_selection)        ──► outputs/selection/*, selected_models.json
src/integration (alignment, inner ROI, object_distance, detector_depth_fusion.PerceptionPipeline)
src/navigation  (ROI, sektor, blocking, arah)
src/inference   (predict_image, predict_video, webcam) + app/main.py (antarmuka Gradio)
src/evaluation/evaluate_{distance,navigation}        ──► outputs/integration/*
src/utils/report                                     ──► outputs/reports/experiment_summary
```

### Pemetaan terhadap struktur yang disarankan prompt (bagian 17)

Prompt mengizinkan penyesuaian struktur bila ada alasan teknis. Kode ditempatkan di bawah paket `src/` (dan
`models/`, `app/`) karena nama paket tingkat atas generik seperti `datasets`, `utils`, `evaluation`, `training`,
atau `inference` dapat bertabrakan dengan paket pip yang terpasang (mis. `datasets` milik Hugging Face) dan
dengan folder data `datasets/` yang tidak di-commit. Pemetaannya:

| Struktur disarankan | Lokasi di repository ini |
|---|---|
| `configs/data/sunrgbd.yaml` | `configs/dataset/sunrgbd.yaml` |
| `configs/detection/{teacher_yolo26x,student_yolo26n,kd_feature,kd_prediction,ca_prediction,ca_feature_prediction}.yaml` | `configs/detection/*.yaml`: 10 berkas DET-01…10 (lihat EXPERIMENT_MATRIX.md) |
| `configs/depth/*.yaml` | `configs/depth/*.yaml`: 14 berkas DEP-01…14 |
| `configs/evaluation/{detection_metrics,depth_metrics,saw_ewm}.yaml` | `configs/evaluation/{detection,depth,selection,distance,navigation}.yaml` |
| `configs/navigation/navigation.yaml` | `configs/system/inference.yaml` (bagian `distance`, `fusion`, `navigation`) |
| `datasets/scripts/*` | `src/data/{download_dataset,prepare_sunrgbd,prepare_depth,prepare_annotations,pair_rgb_depth,create_split,validate_dataset}.py` |
| `models/detection/coordinate_attention.py`, `models/depth/cbam.py` | sama |
| `models/{detection,depth}/yolo26_*_{teacher,student}.py` | `models/detection/yolo26_detector.py`, `models/depth/yolo26_depth.py` (teacher dan student memakai builder yang sama, beda skala `n`/`x`) |
| `models/common/{model_loader,model_inspection,parameter_counter}.py` | `models/common/{model_utils,feature_hooks}.py` |
| `distillation/detection/*` | `models/detection/detection_teacher_student.py`, `src/losses/{feature_loss,detection_kd}.py` |
| `distillation/depth/*` | `models/depth/depth_teacher_student.py`, `src/losses/{depth_output_loss,feature_loss,boundary_loss,relative_depth_loss,combined_loss}.py` |
| `distillation/common/{feature_hooks,feature_alignment,loss_utils}.py` | `models/common/feature_hooks.py`, `src/losses/feature_loss.py::build_feature_projector`, `src/losses/depth_output_loss.py::masked_mean` |
| `training/*.py`, `resume_training.py` | `src/training/train_{detection,detection_kd,depth,depth_kd}.py` (opsi `--resume`), `trainer_utils.py` |
| `evaluation/{detection,depth}/*` | `src/evaluation/{evaluate_detection,evaluate_depth,metrics}.py` |
| `evaluation/efficiency/*` | `src/evaluation/benchmark.py`, `models/common/model_utils.py::model_stats` |
| `evaluation/selection/*` | `src/optimization/{entropy_weight,simple_additive_weighting,model_selection}.py` |
| `evaluation/integration/*` | `src/evaluation/{evaluate_distance,evaluate_navigation}.py`, `src/utils/report.py` |
| `inference/{predict_image,predict_video,webcam,inference_pipeline}.py` | `src/inference/{predict_image,predict_video,webcam,common}.py`, `src/integration/detector_depth_fusion.py` |
| `integration/*` | `src/integration/{detector_depth_fusion,spatial_alignment,inner_roi,object_distance}.py` |
| `navigation/*` | `src/navigation/{navigation_roi,sector_division,obstacle_analysis,direction_decision}.py`, `src/utils/visualization.py` |
| `interface/app.py`, `interface/components/` | `app/main.py`, `app/components/panels.py` |
| `utils/*` | `src/utils/{config,logger,seed,io,checkpoint,visualization,report}.py` |

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

## Model ber-attention (`models/common/model_utils.py::attention_yaml`)

Dipakai oleh DET-06 (YOLO26x + CA), DET-07 (YOLO26n + CA), DEP-08 (YOLO26x-Depth + CBAM), dan DEP-09
(YOLO26n-Depth + CBAM). Tabel berikut untuk skala x; skala n sama dengan kanal lebih kecil.

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
