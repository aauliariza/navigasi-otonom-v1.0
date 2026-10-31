"""EWM-weighted SAW model selection for detection and depth, computed separately (prompt section 8).

Reads the decision matrix produced by src/evaluation/evaluate_{detection,depth}.py (only rows with
``role == alternative``), applies EWM (src/optimization/entropy_weight.py) and SAW
(src/optimization/simple_additive_weighting.py) and writes, per task, under ``output_dir``:

* ``decision_matrix.csv``  raw criterion values
* ``ewm_weights.csv``      p-entropy-diversification-weight per criterion (+ ``ewm_proportions.csv``)
* ``normalized_matrix.csv`` SAW r_ij
* ``saw_ranking.csv``      weighted contributions w_j r_ij, V_i and rank
* ``selection_report.json`` everything above + selected model and the selected KD student for integration
* ``selection_report.md``  human-readable summary

Finally ``outputs/model_selection/selected_models.json`` names the checkpoints used by the integrated system.

Usage::

    python -m src.optimization.model_selection --config configs/evaluation/selection.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from src.optimization.entropy_weight import entropy_weights
from src.optimization.simple_additive_weighting import rank, saw_normalize, saw_scores
from src.utils.config import apply_overrides, load_config, resolve_path
from src.utils.io import load_csv, save_csv, save_json
from src.utils.logger import get_logger

LOG = get_logger("model_selection")
TEACHER_PATTERN = re.compile(r"^YOLO26x", re.IGNORECASE)


def build_matrix(rows: list[dict], criteria: list[dict], memory_column: str, missing_policy: str):
    """Return (names, X, used_criteria, dropped) from evaluation rows."""
    used, dropped = [], []
    for c in criteria:
        col = memory_column if c["column"] == "gpu_memory_gb" else c["column"]
        vals = [r.get(col, "") for r in rows]
        if any(v in ("", None, "None", "nan") for v in vals):
            if missing_policy == "error":
                raise ValueError(f"Criterion '{col}' has missing values")
            dropped.append({"column": col, "reason": "missing value for at least one model (e.g. GPU memory on CPU)"})
            continue
        used.append({**c, "column": col})
    X = np.array([[float(r[c["column"]]) for c in used] for r in rows], dtype=np.float64)
    return [r["model"] for r in rows], X, used, dropped


def select(task: str, cfg: dict, global_cfg: dict) -> dict:
    """Run EWM + SAW for one task and write its reports."""
    rows = [r for r in load_csv(resolve_path(cfg["results"])) if r.get("role", "alternative") == "alternative"]
    if len(rows) < 2:
        raise ValueError(f"{task}: need at least two evaluated alternatives, found {len(rows)}")
    names, X, crit, dropped = build_matrix(rows, cfg["criteria"], global_cfg["memory_column"], global_cfg["missing_policy"])
    benefit = [c["type"] == "benefit" for c in crit]
    R = saw_normalize(X, benefit)
    ewm = entropy_weights(R if global_cfg.get("ewm_basis", "raw") == "normalized" else X)
    w = ewm["weights"]
    V = saw_scores(R, w)
    ranks = rank(V)
    cols = [c["column"] for c in crit]
    out = resolve_path(cfg["output_dir"])
    save_csv([{"model": n, **dict(zip(cols, x))} for n, x in zip(names, X)], out / "decision_matrix.csv")
    save_csv([{"model": n, **dict(zip(cols, p))} for n, p in zip(names, ewm["p"])], out / "ewm_proportions.csv")
    save_csv([{"criterion": c["column"], "type": c["type"], "entropy": e, "diversification": d, "weight": wj}
              for c, e, d, wj in zip(crit, ewm["entropy"], ewm["diversification"], w)], out / "ewm_weights.csv")
    save_csv([{"model": n, **dict(zip(cols, r))} for n, r in zip(names, R)], out / "normalized_matrix.csv")
    ranking = [{"model": n, **{f"w*r[{c}]": v for c, v in zip(cols, r * w)}, "saw_score": s, "rank": int(k)}
               for n, r, s, k in zip(names, R, V, ranks)]
    save_csv(sorted(ranking, key=lambda x: x["rank"]), out / "saw_ranking.csv")

    best = names[int(np.argmax(V))]
    pool = [i for i, n in enumerate(names) if not TEACHER_PATTERN.match(n) and "KD" in n] if global_cfg.get("integration_pool") == "kd_students" else list(range(len(names)))
    if not pool:
        raise ValueError(f"{task}: no KD student among alternatives for integration")
    student = names[max(pool, key=lambda i: V[i])]
    weights_by_model = {r["model"]: r.get("weights") for r in rows}
    report = {"task": task, "alternatives": names, "criteria": crit, "dropped_criteria": dropped, "ewm_basis": global_cfg.get("ewm_basis", "raw"),
              "ewm": {k: v for k, v in ewm.items() if k != "p"}, "saw_scores": dict(zip(names, V)), "ranks": dict(zip(names, ranks.tolist())),
              "selected_overall": best, "selected_kd_student": student, "selected_kd_student_weights": weights_by_model.get(student)}
    save_json(report, out / "selection_report.json")
    _markdown(report, cols, X, R, w, out / "selection_report.md")
    LOG.info("%s: best=%s | KD student for integration=%s", task, best, student)
    return report


def _markdown(rep: dict, cols, X, R, w, path: Path) -> None:
    lines = [f"# Pemilihan model {rep['task']} (EWM-SAW)", "", "## Bobot EWM", "", "| Kriteria | Tipe | Entropy | d | Bobot |", "|---|---|---|---|---|"]
    for c, e, d, wj in zip(rep["criteria"], rep["ewm"]["entropy"], rep["ewm"]["diversification"], w):
        lines.append(f"| {c['column']} | {c['type']} | {e:.6f} | {d:.6f} | {wj:.6f} |")
    lines += ["", "## Matriks keputusan dan normalisasi", "", "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for n, x, r in zip(rep["alternatives"], X, R):
        lines.append(f"| {n} | " + " | ".join(f"{a:.4g} ({b:.3f})" for a, b in zip(x, r)) + " |")
    lines += ["", "Nilai di dalam kurung = r_ij.", "", "## Nilai SAW dan ranking", "", "| Rank | Model | V |", "|---|---|---|"]
    for n in sorted(rep["alternatives"], key=lambda k: rep["ranks"][k]):
        lines.append(f"| {rep['ranks'][n]} | {n} | {rep['saw_scores'][n]:.6f} |")
    lines += ["", f"**Model terpilih (keseluruhan):** {rep['selected_overall']}", "",
              f"**Student KD terpilih untuk integrasi:** {rep['selected_kd_student']}", ""]
    if rep["dropped_criteria"]:
        lines += ["**Kriteria tidak dipakai:** " + ", ".join(d["column"] for d in rep["dropped_criteria"]), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="EWM-SAW model selection")
    ap.add_argument("--config", default="configs/evaluation/selection.yaml")
    ap.add_argument("--task", choices=["detection", "depth", "both"], default="both")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    reports = {t: select(t, cfg[t], cfg) for t in (["detection", "depth"] if args.task == "both" else [args.task])}
    if len(reports) == 2:
        sel = {t: {"model": r["selected_kd_student"], "weights": r["selected_kd_student_weights"]} for t, r in reports.items()}
        path = resolve_path(cfg.get("selected_models_path", "outputs/model_selection/selected_models.json"))
        save_json(sel, path)
        LOG.info("Selected models for integration written to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
