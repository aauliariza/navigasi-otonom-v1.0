"""The experiment configs must match the researcher's list exactly: DET-01..DET-10 and DEP-01..DEP-14."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import load_config

TX, TCA = "configs/detection/teacher_x.yaml", "configs/detection/teacher_x_ca.yaml"
DX, DCB = "configs/depth/teacher_x.yaml", "configs/depth/teacher_x_cbam.yaml"
ALL = {"output", "feature", "boundary", "relative"}

# id: (config, arch, attention, teacher config or None, enabled KD components)
DETECTION = {
    "DET-01": ("teacher_x.yaml", "yolo26x.yaml", "none", None, set()),
    "DET-02": ("baseline.yaml", "yolo26n.yaml", "none", None, set()),
    "DET-03": ("feature_kd.yaml", "yolo26n.yaml", "none", TX, {"feature"}),
    "DET-04": ("prediction_kd.yaml", "yolo26n.yaml", "none", TX, {"prediction"}),
    "DET-05": ("feature_prediction_kd.yaml", "yolo26n.yaml", "none", TX, {"feature", "prediction"}),
    "DET-06": ("teacher_x_ca.yaml", "yolo26x.yaml", "ca", None, set()),
    "DET-07": ("student_n_ca.yaml", "yolo26n.yaml", "ca", None, set()),
    "DET-08": ("ca_feature_kd.yaml", "yolo26n.yaml", "none", TCA, {"feature"}),
    "DET-09": ("ca_prediction_kd.yaml", "yolo26n.yaml", "none", TCA, {"prediction"}),
    "DET-10": ("ca_feature_prediction_kd.yaml", "yolo26n.yaml", "none", TCA, {"feature", "prediction"}),
}
DEPTH = {
    "DEP-01": ("teacher_x.yaml", "yolo26x-depth.yaml", "none", None, set()),
    "DEP-02": ("baseline.yaml", "yolo26n-depth.yaml", "none", None, set()),
    "DEP-03": ("output_kd.yaml", "yolo26n-depth.yaml", "none", DX, {"output"}),
    "DEP-04": ("feature_kd.yaml", "yolo26n-depth.yaml", "none", DX, {"feature"}),
    "DEP-05": ("boundary_kd.yaml", "yolo26n-depth.yaml", "none", DX, {"boundary"}),
    "DEP-06": ("relative_kd.yaml", "yolo26n-depth.yaml", "none", DX, {"relative"}),
    "DEP-07": ("full_kd.yaml", "yolo26n-depth.yaml", "none", DX, ALL),
    "DEP-08": ("teacher_x_cbam.yaml", "yolo26x-depth.yaml", "cbam", None, set()),
    "DEP-09": ("student_n_cbam.yaml", "yolo26n-depth.yaml", "cbam", None, set()),
    "DEP-10": ("cbam_output_kd.yaml", "yolo26n-depth.yaml", "none", DCB, {"output"}),
    "DEP-11": ("cbam_feature_kd.yaml", "yolo26n-depth.yaml", "none", DCB, {"feature"}),
    "DEP-12": ("cbam_boundary_kd.yaml", "yolo26n-depth.yaml", "none", DCB, {"boundary"}),
    "DEP-13": ("cbam_relative_kd.yaml", "yolo26n-depth.yaml", "none", DCB, {"relative"}),
    "DEP-14": ("cbam_full_kd.yaml", "yolo26n-depth.yaml", "none", DCB, ALL),
}


@pytest.mark.parametrize(("task", "table"), [("detection", DETECTION), ("depth", DEPTH)])
def test_experiment_matrix(task, table):
    seen = {}
    for f in sorted(Path(f"configs/{task}").glob("*.yaml")):
        if not f.name.startswith("_"):
            seen[load_config(f)["experiment"]["id"]] = f.name
    assert seen == {k: v[0] for k, v in table.items()}  # exactly these IDs, one config each
    for eid, (name, arch, att, teacher, comps) in table.items():
        c = load_config(f"configs/{task}/{name}")
        kd = c.get("kd", {})
        assert c["model"]["arch"] == arch and c["model"]["attention"] == att, eid
        assert (kd.get("teacher_config") if kd.get("enabled") else None) == teacher, eid
        on = {k for k in ("feature", "prediction", "output", "boundary", "relative") if isinstance(kd.get(k), dict) and kd[k].get("enabled")}
        assert on == (comps if kd.get("enabled") else set()), eid
        x = arch.startswith("yolo26x")
        assert c["train"]["batch"] == (8 if x else 16), eid  # Tabel 3.7 / 3.8
        if task == "depth":
            assert c["train"]["lr0"] == (0.0005 if x else 0.001), eid


def test_teachers_are_trained_before_their_students():
    """Every KD teacher config is itself a non-KD experiment of the matrix."""
    for table in (DETECTION, DEPTH):
        teachers = {t for *_, t, _ in table.values() if t}
        for t in teachers:
            assert not load_config(t).get("kd", {}).get("enabled", False)


def test_evaluation_lists_cover_all_experiments():
    for task, table in (("detection", DETECTION), ("depth", DEPTH)):
        ev = load_config(f"configs/evaluation/{task}.yaml")
        ids = {load_config(m["config"])["experiment"]["id"] for m in ev["models"]}
        assert ids == set(table) and not ev.get("extra_models")


def test_kd_student_ids_for_integration():
    from src.optimization.model_selection import KD_STUDENT_ID

    kd = {eid for table in (DETECTION, DEPTH) for eid, v in table.items() if v[3]}
    for table in (DETECTION, DEPTH):
        for eid in table:
            assert bool(KD_STUDENT_ID.match(eid)) == (eid in kd), eid
