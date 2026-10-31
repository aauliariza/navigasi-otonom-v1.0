"""Detection / depth / distance / navigation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    DepthMetricAccumulator,
    box_iou,
    distance_metrics,
    f1_score,
    match_detections,
    navigation_accuracy,
    precision_recall_f1,
)


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], float)
    assert box_iou(a, b)[0] == pytest.approx([1.0, 25 / 175, 0.0])


def test_match_detections_greedy_one_to_one():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    pred = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], float)
    tp, fp, fn = match_detections(pred, np.array([0.9, 0.8, 0.7]), gt, 0.5)
    assert (tp, fp, fn) == (1, 2, 1)
    assert match_detections(np.zeros((0, 4)), np.zeros(0), gt) == (0, 0, 2)
    assert match_detections(pred, np.ones(3), np.zeros((0, 4))) == (0, 3, 0)


def test_precision_recall_f1():
    r = precision_recall_f1(8, 2, 4)
    assert r["precision"] == pytest.approx(0.8) and r["recall"] == pytest.approx(8 / 12)
    assert r["f1"] == pytest.approx(f1_score(0.8, 8 / 12))
    assert precision_recall_f1(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_depth_metrics_known_values():
    gt = np.array([[1.0, 2.0, 4.0, 0.0, 20.0]])
    pred = np.array([[1.0, 2.5, 3.0, 5.0, 5.0]])
    acc = DepthMetricAccumulator(0.001, 10.0)
    acc.update(pred, gt)
    r = acc.compute()
    g, p = np.array([1.0, 2.0, 4.0]), np.array([1.0, 2.5, 3.0])  # invalid gt 0 and >10 excluded
    assert r["abs_rel"] == pytest.approx(np.mean(np.abs(p - g) / g))
    assert r["sq_rel"] == pytest.approx(np.mean((p - g) ** 2 / g))
    assert r["rmse"] == pytest.approx(np.sqrt(np.mean((p - g) ** 2)))
    assert r["rmse_log"] == pytest.approx(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)))
    ratio = np.maximum(p / g, g / p)
    assert r["delta1"] == pytest.approx(np.mean(ratio < 1.25))
    assert r["delta3"] == pytest.approx(np.mean(ratio < 1.25**3))
    assert r["valid_pixels"] == 3


def test_depth_metrics_pooled_vs_per_image():
    acc = DepthMetricAccumulator()
    acc.update(np.array([[2.0]]), np.array([[1.0]]))
    acc.update(np.ones((1, 3)), np.ones((1, 3)))
    r = acc.compute()
    assert r["abs_rel"] == pytest.approx(1.0 / 4)
    assert r["abs_rel_per_image_mean"] == pytest.approx(0.5)


def test_depth_metrics_clamp_non_finite_prediction():
    acc = DepthMetricAccumulator(0.001, 10.0)
    acc.update(np.array([[np.inf, -1.0]]), np.array([[10.0, 0.001]]))
    r = acc.compute()
    assert np.isfinite(list(r.values())).all()


def test_distance_metrics():
    r = distance_metrics([1.1, 2.0, 2.7], [1.0, 2.0, 3.0])
    assert r["mae"] == pytest.approx(0.4 / 3)
    assert r["rmse"] == pytest.approx(np.sqrt((0.01 + 0 + 0.09) / 3))
    assert r["mape"] == pytest.approx(100 * (0.1 + 0 + 0.1) / 3)
    assert distance_metrics([], [])["n"] == 0
    with pytest.raises(ValueError):
        distance_metrics([1.0], [0.0])


def test_navigation_accuracy():
    r = navigation_accuracy(["FORWARD", "STOP", "TURN_LEFT"], ["FORWARD", "TURN_LEFT", "TURN_LEFT"])
    assert r["n_correct"] == 2 and r["accuracy_pct"] == pytest.approx(200 / 3)
