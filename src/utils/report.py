"""Compile the research report (Markdown) from the artefacts that actually exist under an output root.

Every section is filled only from files written by the pipeline; a missing artefact is reported as
"BELUM DIJALANKAN" (not run yet) and a missing metric of a listed experiment ID as ``NOT MEASURED``, never guessed.

Sections 2-3 list every experiment ID of the prompt (DET-01..10, DEP-01..14) for the validation split (used by
EWM-SAW) and the official test split (final evaluation) separately; the same tables are exported as
``experiment_summary_{detection,depth}_{val,test}.csv`` next to the report.

Usage::

    python -m src.utils.report --root outputs --out outputs/reports/experiment_summary/RESEARCH_REPORT.md
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.utils.config import resolve_path
from src.utils.io import load_csv, load_json, save_csv

NOT_RUN = "_BELUM DIJALANKAN — artefak tidak ditemukan._"
NOT_MEASURED = "NOT MEASURED"
DET_IDS = [f"DET-{i:02d}" for i in range(1, 11)]
DEP_IDS = [f"DEP-{i:02d}" for i in range(1, 15)]
DET_COLS = ["mAP50", "mAP50-95", "precision", "recall", "f1", "fps", "params", "gflops", "gpu_memory_gb"]
DEP_COLS = ["delta1", "delta2", "delta3", "abs_rel", "sq_rel", "rmse", "rmse_log", "fps", "params", "gflops", "gpu_memory_gb"]


def experiment_table(root: Path, fname: str, split: str, ids: list[str], cols: list[str]) -> list[dict]:
    """One row per experiment ID; values from ``<...>/<split>/<fname>`` (or rows tagged with that split)."""
    found: dict[str, dict] = {}
    for p in sorted(root.rglob(fname)):
        for r in load_csv(p):
            if r.get("split", p.parent.name) == split and r.get("experiment_id"):
                found[r["experiment_id"]] = r
    rows = []
    for i in ids:
        r = found.get(i)
        rows.append({"experiment_id": i, "model": r["model"] if r else NOT_MEASURED,
                     **{c: (r.get(c) if r and r.get(c) not in (None, "", "None") else NOT_MEASURED) for c in cols}})
    return rows


def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return "nan"
    return f"{f:.4g}" if abs(f) < 1e4 else f"{f:,.0f}"


def _table(rows: list[dict], cols: list[str]) -> str:
    cols = [c for c in cols if rows and c in rows[0]]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def _find(root: Path, name: str) -> Path | None:
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def build_report(root: Path, out_dir: Path | None = None) -> str:
    """Return the Markdown report for all artefacts below ``root``."""
    s = ["# Laporan Hasil Penelitian", "", f"Dibuat otomatis: {datetime.now().astimezone().isoformat(timespec='seconds')}  ",
         f"Sumber artefak: `{root}`", ""]

    s += ["## 1. Integritas dataset", ""]
    p = _find(root, "integrity_report.json")
    if p:
        r = load_json(p)
        s += [f"Status: **{'LULUS' if r['passed'] else 'GAGAL'}** (`{p}`)", "",
              _table([{"split": k, **v} for k, v in r["summary"].items()], ["split", "rgb", "depth", "labels", "valid_pairs", "invalid_data", "bounding_boxes"]), ""]
    else:
        s += [NOT_RUN, ""]

    for title, task, fname, ids, cols in (
        ("2. Evaluasi deteksi objek", "detection", "detection_results.csv", DET_IDS, DET_COLS),
        ("3. Evaluasi monocular depth estimation", "depth", "depth_results.csv", DEP_IDS, DEP_COLS),
    ):
        s += [f"## {title}", ""]
        for split, label in (("val", "dasar EWM-SAW (mode proposal: alltest 5.050; mode internal: validasi internal)"),
                             ("test", "official test, hanya pada split.mode=internal")):
            rows = experiment_table(root, fname, split, ids, cols)
            if out_dir is not None:
                save_csv(rows, out_dir / f"experiment_summary_{task}_{split}.csv")
            s += [f"### Split {split}: {label}", "", _table(rows, ["experiment_id", "model", *cols]), ""]

    s += ["## 4. Pemilihan model EWM-SAW", ""]
    found = False
    for p in sorted(root.rglob("selection_report.md")):
        found = True
        text = p.read_text(encoding="utf-8").replace("\n## ", "\n#### ").replace("# Pemilihan", "### Pemilihan", 1)
        s += [text, ""]
    if not found:
        s += [NOT_RUN, ""]

    s += ["## 5. Akurasi estimasi jarak objek", ""]
    p = _find(root, "distance_summary.json")
    if p:
        r = load_json(p)
        s += [f"Sumber ground truth: **{r['gt_source']}** (`{p}`)", "",
              _table([r], ["objects_total", "objects_valid", "objects_invalid_pred", "objects_invalid_gt", "objects_not_detected", "mae", "rmse", "mape"]), ""]
    else:
        s += [NOT_RUN, ""]

    s += ["## 6. Ketepatan keputusan navigasi", ""]
    p = _find(root, "navigation_summary.json")
    if p:
        r = load_json(p)
        s += [f"Sumber ground truth: **{r['gt_source']}** (`{p}`)", "", _table([r], ["n_total", "n_correct", "accuracy_pct"]), ""]
    else:
        s += [NOT_RUN, ""]

    s += ["## 7. Efisiensi sistem terintegrasi", ""]
    p = _find(root, "benchmark_system.json")
    if p:
        r = load_json(p)
        rows = [{"mode": k, **{x: r[k][x] for x in ("latency_ms_mean", "latency_ms_std", "fps", "gpu_memory_peak_gb")}} for k in ("pipeline_models_only", "end_to_end")]
        hw = r.get("hardware", {})
        s += [f"Perangkat: {hw.get('gpu') or hw.get('cpu', '-')}, torch {hw.get('torch')}, ultralytics {hw.get('ultralytics')}", "",
              _table(rows, ["mode", "latency_ms_mean", "latency_ms_std", "fps", "gpu_memory_peak_gb"]), ""]
    else:
        s += [NOT_RUN, ""]
    return "\n".join(s)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Compile RESEARCH_REPORT.md from pipeline artefacts")
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--out", default="outputs/reports/experiment_summary/RESEARCH_REPORT.md")
    args = ap.parse_args(argv)
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(resolve_path(args.root), out.parent), encoding="utf-8")
    print(f"Report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
