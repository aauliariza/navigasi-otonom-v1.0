"""Entropy Weight Method and Simple Additive Weighting."""

import numpy as np
import pytest

from src.optimization.entropy_weight import entropy_weights
from src.optimization.simple_additive_weighting import rank, saw_normalize, saw_scores


def test_ewm_manual_example():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    r = entropy_weights(X)
    p = np.array([1, 2, 3]) / 6
    e0 = -np.sum(p * np.log(p)) / np.log(3)
    assert r["entropy"][0] == pytest.approx(e0)
    assert r["entropy"][1] == pytest.approx(1.0)  # identical values -> no information
    assert r["weights"].tolist() == pytest.approx([1.0, 0.0])
    assert r["weights"].sum() == pytest.approx(1.0)


def test_ewm_zero_values_and_zero_column():
    X = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
    r = entropy_weights(X)
    assert np.isfinite(r["weights"]).all()
    assert r["entropy"][0] == pytest.approx(0.0)  # all mass on one alternative
    assert r["weights"][1] == pytest.approx(0.0)  # zero column


def test_ewm_all_identical_falls_back_to_equal_weights():
    r = entropy_weights(np.ones((3, 4)))
    assert r["equal_weight_fallback"] and r["weights"].tolist() == pytest.approx([0.25] * 4)


def test_ewm_input_validation():
    with pytest.raises(ValueError):
        entropy_weights(np.ones((1, 3)))
    with pytest.raises(ValueError):
        entropy_weights(np.array([[1.0, -1.0], [1.0, 2.0]]))
    with pytest.raises(ValueError):
        entropy_weights(np.array([[1.0, np.nan], [1.0, 2.0]]))


def test_saw_normalization_benefit_and_cost():
    X = np.array([[0.8, 10.0], [0.4, 5.0], [0.2, 20.0]])
    R = saw_normalize(X, [True, False])
    assert R[:, 0].tolist() == pytest.approx([1.0, 0.5, 0.25])
    assert R[:, 1].tolist() == pytest.approx([0.5, 1.0, 0.25])  # min/x: lowest cost -> 1


def test_saw_zero_handling():
    R = saw_normalize(np.array([[0.0, 0.0], [0.0, 3.0]]), [True, False])
    assert R[:, 0].tolist() == [0.0, 0.0]
    assert R[:, 1].tolist() == [1.0, 0.0]


def test_saw_scores_and_rank():
    R = np.array([[1.0, 0.5], [0.5, 1.0], [0.2, 0.2]])
    V = saw_scores(R, np.array([0.7, 0.3]))
    assert V.tolist() == pytest.approx([0.85, 0.65, 0.2])
    assert rank(V).tolist() == [1, 2, 3]
    assert rank(np.array([0.5, 0.5, 0.1])).tolist() == [1, 2, 3]


def test_model_selection_end_to_end(tmp_path):
    from src.optimization.model_selection import select
    from src.utils.io import load_json, save_csv

    rows = [
        {"model": "YOLO26x", "role": "alternative", "mAP50": 0.9, "fps": 30, "params": 55e6, "gpu_memory_gb": 1.0, "weights": "x.pt"},
        {"model": "YOLO26n", "role": "alternative", "mAP50": 0.8, "fps": 200, "params": 2.4e6, "gpu_memory_gb": 0.2, "weights": "n.pt"},
        {"model": "YOLO26n+FeatureKD", "role": "alternative", "mAP50": 0.85, "fps": 200, "params": 2.4e6, "gpu_memory_gb": 0.2, "weights": "kd.pt"},
        {"model": "YOLO26x+CA", "role": "extra", "mAP50": 0.95, "fps": 25, "params": 56e6, "gpu_memory_gb": 1.1, "weights": "ca.pt"},
    ]
    save_csv(rows, tmp_path / "res.csv")
    cfg = {"results": str(tmp_path / "res.csv"), "output_dir": str(tmp_path / "out"),
           "criteria": [{"column": "mAP50", "type": "benefit"}, {"column": "fps", "type": "benefit"},
                        {"column": "params", "type": "cost"}, {"column": "gpu_memory_gb", "type": "cost"}]}
    rep = select("detection", cfg, {"memory_column": "gpu_memory_gb", "missing_policy": "drop_criterion", "integration_pool": "kd_students"})
    assert rep["alternatives"] == ["YOLO26x", "YOLO26n", "YOLO26n+FeatureKD"]  # extra row excluded
    assert rep["selected_overall"] == "YOLO26n+FeatureKD"
    assert rep["selected_kd_student"] == "YOLO26n+FeatureKD" and rep["selected_kd_student_weights"] == "kd.pt"
    assert load_json(tmp_path / "out/selection_report.json")["ranks"]["YOLO26n+FeatureKD"] == 1
    assert (tmp_path / "out/selection_report.md").is_file()
