"""Spatial alignment, inner ROI and object distance estimation."""

import numpy as np
import pytest

from src.integration.inner_roi import inner_roi
from src.integration.object_distance import estimate_distance
from src.integration.spatial_alignment import scale_boxes_to_depth, scale_intrinsics


def test_spatial_alignment_eq_3_11_3_12():
    b = scale_boxes_to_depth(np.array([[64, 48, 320, 240]]), (480, 640), (160, 160))
    assert b[0].tolist() == pytest.approx([16, 16, 80, 80])
    assert scale_boxes_to_depth(np.array([[-10, 0, 700, 500]]), (480, 640), (160, 160))[0].tolist() == [0, 0, 160, 160]
    assert scale_boxes_to_depth(np.zeros((0, 4)), (10, 10), (5, 5)).shape == (0, 4)


def test_scale_intrinsics():
    K = scale_intrinsics(np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]]), (480, 640), (160, 160))
    assert K[0, 0] == pytest.approx(125) and K[1, 1] == pytest.approx(500 / 3) and K[0, 2] == pytest.approx(80)


def test_inner_roi_60_percent():
    assert inner_roi([0, 0, 100, 50], 0.6) == (20, 10, 80, 40)  # x1+0.2w .. x2-0.2w, y2-0.2h


def test_inner_roi_minimum_size_and_bounds():
    c0, r0, c1, r1 = inner_roi([10, 10, 12, 12], 0.6, 4, (100, 100))
    assert c1 - c0 >= 4 and r1 - r0 >= 4
    c0, r0, c1, r1 = inner_roi([0, 0, 2, 2], 0.6, 4, (100, 100))
    assert (c0, r0) == (0, 0) and c1 - c0 >= 4
    assert inner_roi([98, 98, 100, 100], 0.6, 4, (100, 100))[2:] == (100, 100)


def test_distance_median_and_validity():
    depth = np.full((50, 50), 2.0)
    rec = estimate_distance(depth, [0, 0, 50, 50])
    assert rec["valid"] and rec["distance_m"] == pytest.approx(2.0) and rec["valid_ratio"] == 1.0
    depth[:, :] = 0.0
    depth[:5, :] = 3.0  # inside ROI (rows 10-40) everything is invalid
    rec = estimate_distance(depth, [0, 0, 50, 50])
    assert not rec["valid"] and rec["reason"] == "low_valid_ratio" and rec["distance_m"] is None


def test_distance_valid_ratio_threshold_and_range_filter():
    depth = np.zeros((10, 10))
    depth[:, :3] = 1.5  # 30 % of a full-frame ROI
    rec = estimate_distance(depth, [0, 0, 10, 10], roi_ratio=1.0)
    assert rec["valid_ratio"] == pytest.approx(0.3) and rec["valid"]
    depth[:, 2] = 0.05  # below 0.1 m -> invalid, ratio 0.2
    rec = estimate_distance(depth, [0, 0, 10, 10], roi_ratio=1.0)
    assert rec["valid_ratio"] == pytest.approx(0.2) and not rec["valid"]
    depth[:] = np.nan
    assert not estimate_distance(depth, [0, 0, 10, 10])["valid"]


def test_distance_percentile_clipping_robust_to_outliers():
    depth = np.full((20, 20), 1.0)
    depth[0:2, :] = 9.0  # outliers in ROI? ROI rows 4..16 -> put them inside
    depth[8, 4:16] = 9.5
    rec = estimate_distance(depth, [0, 0, 20, 20])
    assert rec["distance_m"] == pytest.approx(1.0)
    assert rec["p_high"] <= 9.5


def test_euclidean_mode_larger_than_axial_off_centre():
    depth = np.full((100, 100), 2.0)
    K = np.array([[50.0, 0, 50], [0, 50.0, 50], [0, 0, 1]])
    centre = estimate_distance(depth, [45, 45, 55, 55], mode="euclidean", intrinsics=K)["distance_m"]
    corner = estimate_distance(depth, [0, 0, 10, 10], mode="euclidean", intrinsics=K)["distance_m"]
    assert centre == pytest.approx(2.0, rel=5e-3) and corner > 2.5  # ROI pixels are <=3 px off the optical centre
    with pytest.raises(ValueError):
        estimate_distance(depth, [0, 0, 10, 10], mode="euclidean")
