# Source Provenance

Dokumen ini memetakan setiap komponen ke publikasi dan kode sumber yang menjadi dasarnya, berkas yang diadaptasi,
dan modifikasinya. Versi kode yang diperiksa: **`ultralytics==8.4.194`** (paket PyPI, diperiksa langsung di
`site-packages`), repository resmi CA (`Andrew-Qibin/CoordAttention`) dan CBAM (`Jongchan/attention-module`) sebagai
rujukan struktur (tidak disalin; ditulis ulang mengikuti paper dan kode resmi), serta repository referensi
`aauliariza/autonomous-wheelchair-v1.0` (commit `main` terbaru saat audit, 31 Oktober 2026).

## 0. Tabel provenance ringkas

| Komponen | Paper/Publikasi | Repository | File yang Diadaptasi | Modifikasi |
|---|---|---|---|---|
| YOLO26 (DET) | Ultralytics (2026), dokumentasi YOLO26 (dokumentasi teknis, bukan paper peer-review) | ultralytics/ultralytics 8.4.194 | `cfg/models/26/yolo26.yaml`, `nn/modules/head.py::Detect` | Tidak ada; `nc = 1` |
| YOLO26-Depth | Ultralytics, dokumentasi task depth | idem | `cfg/models/26/yolo26-depth.yaml`, `nn/modules/head.py::Depth`, `utils/loss.py::DepthLoss26`, `models/yolo/depth/*` | Arsitektur & loss GT tidak diubah; kalibrasi otomatis diganti log-affine 2 parameter pada validasi internal |
| Feature KD deteksi (DET-03, DET-06) | Proposal 3.2.4.2; Romero et al. (2015) FitNets (konsep); Jung & Nam (2025) | ultralytics 8.4.194 | `nn/distill_model.py` (`DistillationModel`, `loss_sl2`) | Subclass `DetectionKDModel`; logika L2 berbobot skor teacher identik (diuji) |
| Prediction KD deteksi (DET-04…06) | Hinton et al. (2015, arXiv); Chen et al. (NeurIPS 2017) | — (tidak ada untuk YOLO26) | — | **Adaptasi konsep paper, bukan reproduksi kode penulis**: BCE soft-label + L1 box berbobot confidence teacher pada kedua cabang head |
| Coordinate Attention | Hou, Zhou & Feng (CVPR 2021), doi:10.1109/CVPR46437.2021.01350 | Andrew-Qibin/CoordAttention | `coordatt.py` (rujukan struktur) | Ditulis ulang, konstruktor gaya YAML Ultralytics; disisipkan pada P3/P4/P5 teacher |
| CBAM | Woo et al. (ECCV 2018), doi:10.1007/978-3-030-01234-2_1 | Jongchan/attention-module | `MODELS/cbam.py` (rujukan struktur) | Ditulis ulang (channel gate + spatial gate 7×7 + BN); disisipkan pada P3/P4/P5 teacher depth |
| Depth Output KD | Lee et al. (IEEE Access 2025); Girshick (ICCV 2015) untuk Smooth L1 | — | — | **Adaptasi konsep** (rumus Pers. 2.9 proposal) |
| Feature KD depth | Song & Kim (JVCIR 2022); Jung & Nam (2025) | — | — | **Adaptasi konsep**: proyektor teacher→student, MSE fitur ter-normalisasi L2 |
| Boundary KD | Liu & Wang (Electronics 2024) | — | — | **Adaptasi konsep** (Pers. 2.11–2.12) dengan mask gradien valid |
| Relative Depth KD | Zhang et al. (Neurocomputing 2026); Chen et al. (NeurIPS 2016) | — | — | **Adaptasi konsep** (Pers. 2.13–2.14), 4.096 pasangan, toleransi 0,03, sampler ber-seed |
| Kalibrasi log-affine | Jun, Lee & Kim (JVCIR 2024) | ultralytics `models/yolo/depth/calibrate.py` (pembanding) | — | Kuadrat terkecil 2 parameter pada validasi internal |
| Dekode depth SUN RGB-D | Song, Lichtenberg & Xiao (CVPR 2015); proposal Pers. 3.1–3.2 | ultralytics `cfg/datasets/depth-sunrgbd.yaml` (skrip download) | rumus `(d >> 3) \| (d << 13)` | Nilai > 10 m diinvalidasi, split resmi dipakai |
| Split | Pembagian resmi SUN RGB-D (`allsplit.mat`) | SUNRGBDtoolbox | `traintestSUNRGBD/allsplit.mat` | Mode proposal: alltrain → train, alltest → val; mode internal opsional: validasi internal per grup rekaman (implementasi repository ini) |
| EWM | Proposal Pers. 3.29–3.32; Linh, Son & Thao (ETASR 2025) | — | — | Implementasi rumus; kasus 0, konstan, NaN/Inf, negatif ditangani |
| SAW | Proposal Pers. 3.27–3.33 | — | — | Normalisasi cost `min/x` (koreksi Pers. 3.28) |
| Inner ROI & jarak objek | Proposal Pers. 3.11–3.19; Ali & Yu (Sensors 2026) | — | — | Koreksi Pers. 3.17 |
| Navigasi lima sektor | Proposal 3.2.4.7 | — | — | Implementasi literal aturan proposal; tanpa hysteresis/smoothing |
| Pola dataset/KD | — | aauliariza/autonomous-wheelchair-v1.0 (AGPL-3.0) | `datasets/scripts/prepare_sunrgbd.py`, `convert_sunrgbd_obstacle.py` (dibaca, tidak disalin) | Hanya pola desain (normalisasi path scene, format Meta2DBB) |

Seluruh rincian per komponen ada di bagian 1–7 di bawah.

**Arti status**

| Status | Arti |
|---|---|
| **Terverifikasi** | Kode sumber diperiksa langsung (paket `ultralytics==8.4.194` yang terpasang) dan dipakai tanpa perubahan atau dengan pewarisan kelas. |
| **Diadaptasi** | Mengikuti arsitektur atau rumus publikasi dan kode resminya, dengan perubahan antarmuka atau penempatan yang dijelaskan. |
| **Implementasi eksperimental** | Tidak ada implementasi resmi untuk YOLO26. Rumus diturunkan dari proposal dan literatur, dengan parameter yang dapat dikonfigurasi. |

**Catatan verifikasi bibliografi.** Lingkungan pengembangan tidak dapat mengakses `doi.org`, `api.crossref.org`, maupun
`rgbd.cs.princeton.edu` (diblokir oleh proxy). Keberadaan artikel berikut berhasil dicek melalui mesin pencari
publikasi Consensus (Semantic Scholar/OpenAlex):

- Hou et al. (2021), Coordinate Attention.
- Woo et al. (2018), CBAM.
- Jung & Nam (2025), *Lightweight YOLO-Based Real-Time Fall Detection Using Feature Map-Level KD*.

DOI lain ditulis sesuai sumber aslinya (daftar referensi proposal, atau publikasi klasik yang dikenal luas), tetapi
**belum dapat dicek ulang secara daring dari lingkungan ini**. Kolom "Verifikasi bibliografi" menandai hal ini.
Mohon cek ulang sebelum dikutip di tesis.

## 1. Kerangka dan arsitektur dasar

| Komponen | Referensi jurnal / dokumen | Repository | File sumber | Adaptasi untuk penelitian ini | Status |
|---|---|---|---|---|---|
| YOLO26 (deteksi) | Ultralytics (2026), dokumentasi YOLO26, https://docs.ultralytics.com/models/yolo26/ | https://github.com/ultralytics/ultralytics (AGPL-3.0) | `ultralytics/cfg/models/26/yolo26.yaml`, `ultralytics/nn/modules/head.py` (`Detect`) | Arsitektur tidak diubah; `nc=1` (kelas "objek"). | Terverifikasi |
| YOLO26-Depth | Ultralytics, https://docs.ultralytics.com/tasks/depth/ | idem | `cfg/models/26/yolo26-depth.yaml`, `nn/modules/head.py` (`Depth`), `models/yolo/depth/{train,val,predict}.py`, `utils/loss.py` (`DepthLoss26`) | Arsitektur dan loss ground truth (SILog + gradient matching) tidak diubah. Kalibrasi otomatis Ultralytics (scale-only) diganti dengan kalibrasi log-affine dua parameter yang di-fit pada validasi internal (lihat §4). | Terverifikasi |
| Trainer, EMA, early stopping, resume | idem | idem | `ultralytics/engine/trainer.py` | Dipakai apa adanya. Kelas trainer diturunkan (`ResearchTrainerMixin`) untuk membangun model ber-attention, transfer bobot, dan log memori GPU. | Terverifikasi |
| Dataset SUN RGB-D | S. Song, S. P. Lichtenberg, J. Xiao, "SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite", CVPR 2015, doi:10.1109/CVPR.2015.7298655 (verifikasi bibliografi: belum daring) | https://rgbd.cs.princeton.edu/ | `SUNRGBD.zip`, `SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat`, `SUNRGBDMeta2DBB_v2.mat` | Split resmi `alltrain`/`alltest` (5.285/5.050; alltest = test), validasi internal per grup dari alltrain, dekode bit-rotation, anotasi 2D satu kelas. | Diadaptasi |
| Dekode depth SUN RGB-D | Proposal Persamaan (3.1)–(3.2) | Ultralytics `cfg/datasets/depth-sunrgbd.yaml` (skrip `download`) | baris `(d >> 3) \| (d << 13)` | Rumus sama. Bedanya: nilai > 10 m **diinvalidasi**, tidak di-clip (keputusan D-3), dan split resmi dipakai menggantikan split acak 9.245/1.090 milik Ultralytics. | Diadaptasi |
| Format anotasi `SUNRGBDMeta2DBB_v2.mat` | Field `gtBb2D = [x, y, w, h]` | crmauceri/SUNRGBD_COCO (`SUNRGB_to_COCO.m`), facebookresearch/votenet (`sunrgbd/matlab/extract_rgbd_data_v2.m`), seperti terdokumentasi di repository referensi `aauliariza/autonomous-wheelchair-v1.0` (`datasets/scripts/convert_sunrgbd_obstacle.py`) | — | Pembacaan dengan `scipy.io.loadmat`, satu kelas, aturan pembersihan box. Format diuji dengan berkas `.mat` sintetis berstruktur sama, belum dengan berkas asli (lihat bagian *Komponen yang belum dapat diuji* di README). | Diadaptasi |

## 2. Attention

| Metode | Referensi jurnal | Repository | File sumber | Adaptasi | Status |
|---|---|---|---|---|---|
| Coordinate Attention (CA) | Q. Hou, D. Zhou, J. Feng, "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021, pp. 13713–13722, doi:10.1109/CVPR46437.2021.01350 (keberadaan: terverifikasi via Consensus) | https://github.com/Andrew-Qibin/CoordAttention (URL dari abstrak artikel) | `coordatt.py` | Struktur resmi dipertahankan: pooling X/Y, `mip = max(8, C/32)`, BN, h-swish, dua konv 1×1 dengan sigmoid. Konstruktor diubah menjadi baris YAML Ultralytics. Disisipkan pada **teacher YOLO26x** di P3/P4/P5 (layer 23–25) tepat sebelum Detect head (layer 26). | Diadaptasi (dimodifikasi) |
| CBAM | S. Woo, J. Park, J.-Y. Lee, I. S. Kweon, "CBAM: Convolutional Block Attention Module", ECCV 2018, LNCS 11211, pp. 3–19, doi:10.1007/978-3-030-01234-2_1 (keberadaan: terverifikasi via Consensus) | https://github.com/Jongchan/attention-module | `MODELS/cbam.py` | Channel gate (MLP bersama, avg+max pool, reduksi 16), lalu spatial gate (konv 7×7 + BN atas [max, mean]). Disisipkan pada **teacher YOLO26x-Depth** di P3/P4/P5 (layer 23–25) sebelum Depth head (layer 26). | Diadaptasi (dimodifikasi) |

Proposal BAB 1–3 **tidak** mencantumkan CA maupun CBAM. Keduanya diimplementasikan karena diminta secara
eksplisit pada spesifikasi eksperimen (prompt bagian 5). Lihat EXPERIMENT_DESIGN_DECISIONS.md, keputusan A-1.

## 3. Knowledge Distillation deteksi objek

| Metode | Referensi jurnal | Repository | File sumber | Adaptasi | Status |
|---|---|---|---|---|---|
| Kerangka teacher–student (teacher frozen, hook, proyektor, checkpoint student saja) | Proposal Sub-bab 3.2.4.2 ("mekanisme distillation dari framework … *score-weighted* L2 … parameter dis = 6.0") | Ultralytics 8.4.194 | `ultralytics/nn/distill_model.py` (`DistillationModel`, `FeatureHook`, `loss_sl2`); `engine/trainer.py` | Diturunkan menjadi `DetectionKDModel` (`models/detection/detection_teacher_student.py`). Logika Ultralytics dipertahankan. Kelas disisipkan ke trainer lewat substitusi kelas selama training (`distillation_class`), tanpa menyunting source Ultralytics. | Terverifikasi + diperluas |
| Feature KD | Proposal 3.2.4.2; Jung & Nam (2025), ICT Express 11(6):1152–1161, doi:10.1016/j.icte.2025.08.009 (keberadaan: terverifikasi via Consensus); A. Romero et al., "FitNets: Hints for Thin Deep Nets", ICLR 2015, arXiv:1412.6550 (konsep regressor student→teacher) | Ultralytics | `distill_model.py::loss_sl2` | Persis milik Ultralytics: MSE per lokasi antara fitur student terproyeksi dan fitur teacher pada input Detect head (16/19/22 ↔ teacher 16/19/22, atau 23/24/25 untuk teacher CA), diberi bobot oleh probabilitas kelas maksimum teacher, dengan `dis = 6.0`. `src/losses/feature_loss.py::score_weighted_l2` adalah salinan fungsional yang diuji kesetaraannya. | Terverifikasi |
| Prediction KD | G. Hinton, O. Vinyals, J. Dean, "Distilling the Knowledge in a Neural Network", arXiv:1503.02531 (2015); G. Chen et al., "Learning Efficient Object Detection Models with Knowledge Distillation", NeurIPS 2017; Yao et al. (2025), CMC 83(2):2189–2205, doi:10.32604/cmc.2025.060609 (dari proposal, belum diverifikasi daring) | — (tidak ada implementasi YOLO26) | `src/losses/detection_kd.py` | (a) Soft-label BCE kelas `BCE(σ(s_S/T), σ(s_T/T))·T²` pada semua anchor, dinormalisasi Σ confidence teacher. (b) L1 jarak box (ltrb, `reg_max=1`) berbobot confidence teacher. Rata-rata cabang one2many dan one2one. | Implementasi eksperimental |

## 4. Knowledge Distillation monocular depth

| Metode | Referensi jurnal | Repository | File sumber | Adaptasi | Status |
|---|---|---|---|---|---|
| Kerangka teacher–student depth | Proposal 3.2.4.4 | Ultralytics `distill_model.py` | `models/depth/depth_teacher_student.py` | `DepthKDModel(DistillationModel)`: tap = input Depth head (student 16/19/22 sesuai proposal), proyektor teacher→student, loss gabungan. | Diadaptasi |
| Depth Output Distillation (Pers. 2.9) | Lee et al. (2025), IEEE Access 13:2763–2782, doi:10.1109/ACCESS.2024.3523912; Choi, Choi & Kim (2024), IVC 148:105110, doi:10.1016/j.imavis.2024.105110; Smooth L1: R. Girshick, "Fast R-CNN", ICCV 2015, doi:10.1109/ICCV.2015.169 (verifikasi bibliografi: belum daring) | — | `src/losses/depth_output_loss.py` | SmoothL1 (β = 0,1) pada residual log-depth, rata-rata atas mask valid (GT 0,001–10 m dan teacher finite > 0). | Implementasi eksperimental (rumus proposal) |
| Feature Distillation (Pers. 2.10) | Song & Kim (2022), JVCIR 85:103523, doi:10.1016/j.jvcir.2022.103523; Jung & Nam (2025) | — | `src/losses/feature_loss.py` | Proyektor 1×1 dua lapis teacher→student pada layer 16/19/22; `d` = MSE antar fitur yang dinormalisasi L2 per kanal. | Implementasi eksperimental |
| Boundary Distillation (Pers. 2.11–2.12) | Liu & Wang (2024), Electronics 13(9):1652, doi:10.3390/electronics13091652; Yang et al. (2025), Neurocomputing 640:130314 (dari proposal) | — | `src/losses/boundary_loss.py` | Beda maju (*forward difference*), magnitudo `√(Gx²+Gy²)`, SmoothL1. Gradien hanya valid bila semua piksel yang dipakai valid. | Implementasi eksperimental |
| Relative Depth Distillation (Pers. 2.13–2.14) | Zhang et al. (2026), Neurocomputing 671:132632, doi:10.1016/j.neucom.2026.132632; W. Chen, Z. Fu, D. Yang, J. Deng, "Single-Image Depth Perception in the Wild", NeurIPS 2016, arXiv:1604.03901 (bentuk *ranking loss* dan suku pasangan "sama") | — | `src/losses/relative_depth_loss.py` | 4.096 pasangan per citra disampel dengan `torch.Generator` ber-seed. Label ordinal dengan toleransi 0,03. Pasangan `r≠0` memakai logistic ranking (Pers. 2.14). Pasangan `r=0` memakai `ΔS²` (Chen et al., 2016), yang dapat diganti `ignore`. | Implementasi eksperimental |
| Total loss (Pers. 2.8 / prompt 5.2-F) | Proposal Pers. (3.10), tanpa suku ROI | — | `src/losses/combined_loss.py` | λ_output 0,5; λ_feature 0,2; λ_boundary 0,1; λ_relative 0,1. Setiap komponen dapat di-*toggle* dan dicatat. | Diadaptasi |
| Kalibrasi log-affine (Pers. 2.4) | Jun, Lee & Kim (2024), JVCIR 103:104252, doi:10.1016/j.jvcir.2024.104252 | Ultralytics `models/yolo/depth/calibrate.py` (sebagai pembanding) | `models/depth/yolo26_depth.py` | Kuadrat terkecil bentuk tertutup `log D_gt = a·log D_pred + b` pada piksel valid split **val** (validasi internal dari alltrain; test tidak pernah dipakai). Nilai (a, b) ditulis ke buffer `cal_a`/`cal_b` checkpoint. Bila `a ≤ 0`, dipakai fallback scale-only. | Diadaptasi |

Object ROI Distillation dari proposal (Pers. 3.9–3.10) **tidak diimplementasikan**, sesuai instruksi.

## 5. Evaluasi, pemilihan model, integrasi, navigasi

| Metode | Referensi | File | Catatan | Status |
|---|---|---|---|---|
| mAP50, mAP50-95 | Proposal Pers. 3.21; Ultralytics `DetectionValidator` | `src/evaluation/evaluate_detection.py` | Satu kelas: mAP = AP. | Terverifikasi |
| Precision, Recall, F1 | Proposal Pers. 3.22–3.24 | `src/evaluation/metrics.py` | Pencocokan greedy satu-ke-satu pada IoU ≥ 0,5 untuk deteksi conf ≥ 0,25. Nilai P/R Ultralytics pada F1-maks juga disimpan (`*_maxf1`). | Diadaptasi |
| Metrik depth (δ1–3, AbsRel, SqRel, RMSE, RMSE log) | Proposal Pers. 3.34–3.38; D. Eigen, C. Puhrsch, R. Fergus, NeurIPS 2014, arXiv:1406.2283 | `src/evaluation/metrics.py` | Dihitung *pooled* atas semua piksel valid (utama), plus rata-rata per citra. Tanpa median alignment. | Diadaptasi |
| FLOPs, FPS, memori | Proposal Pers. 3.25–3.26; Liu et al. (2022), IEEE Access 10:39674 (memori GPU) | `src/evaluation/benchmark.py` | FLOPs = 2×MAC (`thop.profile` dari `ultralytics-thop`, dipanggil langsung agar error tidak menjadi 0; nilai sama dengan `get_flops`). | Terverifikasi |
| EWM | Proposal Pers. 3.29–3.32; Linh, Son & Thao (2025), ETASR 15(1):19998–20004, doi:10.48084/etasr.9518 | `src/optimization/entropy_weight.py` | Kasus nilai nol, kolom identik, dan m = 1 ditangani eksplisit. | Diadaptasi |
| SAW | Proposal Pers. 3.27, 3.28, 3.33 | `src/optimization/simple_additive_weighting.py` | Normalisasi cost memakai `min/x` (koreksi Pers. 3.28, keputusan M-2). | Diadaptasi |
| Inner ROI, jarak objek | Proposal Pers. 3.11–3.19; Ali & Yu (2026), Sensors 26(16):5013, doi:10.3390/s26165013 (protokol *same-box*) | `src/integration/*` | Pers. 3.17 dikoreksi menjadi `y2 − 0,2h` (keputusan I-1). | Diadaptasi |
| Navigasi ruang bebas | Proposal Sub-bab 3.2.4.7; Machkour, Ortiz-Arroyo & Durdevic (2023), IJCIS 16:79, doi:10.1007/s44196-023-00250-5 | `src/navigation/*` | 5 sektor, prioritas CTR→L→R→FL→FR, tanpa path planning. | Diadaptasi |

## 6. Repository referensi tambahan

`https://github.com/aauliariza/autonomous-wheelchair-v1.0` (AGPL-3.0, karya peneliti sebelumnya) diaudit
(lihat `docs/AUDIT.md`). Tidak ada berkas yang disalin utuh. Bagian yang dipakai sebagai rujukan desain:

- Validasi jalur split resmi SUN RGB-D: `scene_relpath`, pencocokan bagian path setelah `SUNRGBD/`.
- Dokumentasi format `SUNRGBDMeta2DBB_v2.mat`.
- Pola *masked mean* yang mengembalikan nol yang tetap terhubung ke graph.

Bagian yang **tidak** dipakai, karena di luar proposal atau tidak sesuai instruksi:

- ROI-KD.
- Hysteresis/majority vote.
- `EMERGENCY_STOP` latch dan safety layer.
- Optuna tuning.
- Registry global proyektor yang tidak ikut dioptimasi.

## 7. Lisensi

Ultralytics berlisensi **AGPL-3.0**. Repository ini memakai dan memperluas kelasnya, sehingga dirilis di bawah
AGPL-3.0 (lihat `LICENSE`). Kode CA dan CBAM ditulis ulang mengikuti publikasi. Lisensi repository resminya perlu
dicek pada repository masing-masing.
