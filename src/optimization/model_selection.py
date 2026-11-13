"""EWM-weighted SAW model selection for detection and depth, computed separately (prompt section 8).

Reads the decision matrix produced by src/evaluation/evaluate_{detection,depth}.py (only rows with
``role == alternative``), applies EWM (src/optimization/entropy_weight.py) and SAW
(src/optimization/simple_additive_weighting.py) and writes, per task (``<task>`` = detection | depth):

* ``<output_dir>/<task>_ewm/decision_matrix.csv``  raw criterion values
* ``<output_dir>/<task>_ewm/ewm_proportions.csv``  p_ij (Eq. 3.29)
* ``<output_dir>/<task>_ewm/ewm_weights.csv``      entropy, diversification, weight per criterion
* ``<output_dir>/<task>_saw/normalized_matrix.csv`` SAW r_ij (Eq. 3.27-3.28)
* ``<output_dir>/<task>_saw/saw_scores.csv``       weighted contributions w_j r_ij and V_i (Eq. 3.33)
* ``<output_dir>/<task>_saw/ranking.csv``          rank of every configuration
* ``<output_dir>/<task>_saw/ranking.png``          bar chart of V_i
* ``<output_dir>/<task>_saw/selection_report.{json,md}`` everything above + best model / selected KD student

Only results of the VALIDATION split are accepted (``split`` column written by the evaluation scripts); results
of the official test split are refused unless ``allow_test_split: true`` (test data must not drive selection).
Finally ``selected_models_path`` (default ``outputs/selection/selected_models.json``) names the checkpoints used
by the integrated system.

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
# Experiment IDs of the KD students: DET-03..05, DET-08..10 and DEP-03..07, DEP-10..14 (DET-06/07 and DEP-08/09 are
# attention models trained without KD; DET-01/02 and DEP-01/02 are the teacher and the baseline).
KD_STUDENT_ID = re.compile(r"^(DET-0[3-5]|DET-0[89]|DET-10|DEP-0[3-7]|DEP-1[0-4])$")


def is_kd_student(row: dict) -> bool:
    """KD student by experiment ID (preferred) or, for legacy rows without an ID, by display name."""
    eid = row.get("experiment_id") or ""
    if eid:
        return bool(KD_STUDENT_ID.match(eid))
    return not TEACHER_PATTERN.match(row["model"]) and "KD" in row["model"]


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
    splits = {r.get("split", "") for r in rows}
    if "test" in splits and not global_cfg.get("allow_test_split", False):
        raise ValueError(f"{task}: {cfg['results']} contains results of the TEST split. Model selection must use the "
                         "validation split (evaluate with split=val). Set allow_test_split=true only for a documented analysis.")
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
    base = resolve_path(cfg["output_dir"])
    ewm_dir, out = base / f"{task}_ewm", base / f"{task}_saw"
    save_csv([{"model": n, **dict(zip(cols, x))} for n, x in zip(names, X)], ewm_dir / "decision_matrix.csv")
    save_csv([{"model": n, **dict(zip(cols, p))} for n, p in zip(names, ewm["p"])], ewm_dir / "ewm_proportions.csv")
    save_csv([{"criterion": c["column"], "type": c["type"], "entropy": e, "diversification": d, "weight": wj}
              for c, e, d, wj in zip(crit, ewm["entropy"], ewm["diversification"], w)], ewm_dir / "ewm_weights.csv")
    save_csv([{"model": n, **dict(zip(cols, r))} for n, r in zip(names, R)], out / "normalized_matrix.csv")
    ranking = [{"model": n, **{f"w*r[{c}]": v for c, v in zip(cols, r * w)}, "saw_score": s, "rank": int(k)}
               for n, r, s, k in zip(names, R, V, ranks)]
    ranking = sorted(ranking, key=lambda x: x["rank"])
    save_csv(ranking, out / "saw_scores.csv")
    save_csv([{"rank": r["rank"], "model": r["model"], "saw_score": r["saw_score"]} for r in ranking], out / "ranking.csv")
    plot_ranking(ranking, task, out / "ranking.png")

    best = names[int(np.argmax(V))]
    pool = [i for i, r in enumerate(rows) if is_kd_student(r)] if global_cfg.get("integration_pool") == "kd_students" else list(range(len(names)))
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


def plot_ranking(ranking: list[dict], task: str, path: Path) -> None:
    """Horizontal bar chart of SAW scores (best at the top)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["model"] for r in ranking][::-1]
    vals = [float(r["saw_score"]) for r in ranking][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.2), tight_layout=True)
    bars = ax.barh(names, vals, color=["#2a6fdb" if i == len(names) - 1 else "#9bb7e3" for i in range(len(names))])
    for b, v in zip(bars, vals):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Nilai SAW (V_i)")
    ax.set_title(f"Ranking EWM-SAW: {task}")
    ax.set_xlim(0, max(vals) * 1.15 if vals and max(vals) > 0 else 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
        path = resolve_path(cfg.get("selected_models_path", "outputs/selection/selected_models.json"))
        save_json(sel, path)
        LOG.info("Selected models for integration written to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
