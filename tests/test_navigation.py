"""Global ROI, five sectors, blocking rules and direction priority."""

import pytest

from src.navigation.direction_decision import decide_direction
from src.navigation.navigation_roi import global_navigation_roi
from src.navigation.obstacle_analysis import BlockingRules, analyse_sectors, blocking_reason, overlap_ratio
from src.navigation.sector_division import divide_sectors

W, H = 1000, 500


def _sectors():
    roi = global_navigation_roi(W, H)
    return roi, divide_sectors(roi)


def test_global_roi_and_sector_widths():
    roi, sec = _sectors()
    assert roi == (200.0, 0.0, 800.0, 500.0)
    assert [s["name"] for s in sec] == ["FL", "L", "CTR", "R", "FR"]
    assert all(s["x2"] - s["x1"] == pytest.approx(0.12 * W) for s in sec)
    assert sec[0]["x1"] == 200 and sec[-1]["x2"] == 800


def test_overlap_ratio_eq_3_44():
    _, sec = _sectors()
    ctr = sec[2]  # x 440..560
    assert overlap_ratio([440, 0, 560, 100], ctr) == pytest.approx(1.0)
    assert overlap_ratio([400, 0, 500, 100], ctr) == pytest.approx(0.6)
    assert overlap_ratio([0, 0, 100, 100], ctr) == 0.0
    assert overlap_ratio([5, 5, 5, 10], ctr) == 0.0


def _obj(x1, x2, dist=0.5, valid=True, conf=0.9):
    return {"bbox": [x1, 0, x2, 100], "distance_m": dist if valid else None, "valid": valid, "confidence": conf, "fusion_confidence": conf}


def test_blocking_reasons():
    r = BlockingRules()
    assert blocking_reason(_obj(0, 1, 0.5), r) == "near"
    assert blocking_reason(_obj(0, 1, 1.5), r) is None
    assert blocking_reason(_obj(0, 1, 1.0), r) is None  # strictly < 1.0 m
    assert blocking_reason(_obj(0, 1, valid=False), r) == "invalid_depth"
    assert blocking_reason(_obj(0, 1, valid=False), BlockingRules(invalid_depth_blocks=False)) is None
    assert blocking_reason(_obj(0, 1, 3.0, conf=0.1), r) == "low_confidence"
    assert blocking_reason(_obj(0, 1, 3.0, conf=0.1), BlockingRules(low_confidence_blocks=False)) is None


def test_min_overlap_and_objects_outside_roi():
    roi, sec = _sectors()
    res = analyse_sectors([_obj(0, 150), _obj(436, 450)], sec, roi, BlockingRules())
    assert not res[0]["blocked"]  # first object (x 0-150) lies outside the ROI and influences nothing
    # second object: overlap with L = (440-436)/14 = 0.29 -> L blocked; CTR overlap 10/14 -> blocked
    assert res[1]["blocked"] and res[2]["blocked"]
    res = analyse_sectors([_obj(430, 800)], sec, roi, BlockingRules(min_overlap=0.05))
    assert not res[1]["blocked"]  # overlap with L = 10/370 < 0.05


@pytest.mark.parametrize(
    "blocked,expected",
    [
        ({}, ("FORWARD", "CTR")),
        ({"CTR"}, ("TURN_LEFT", "L")),
        ({"CTR", "L"}, ("TURN_RIGHT", "R")),
        ({"CTR", "L", "R"}, ("TURN_LEFT", "FL")),
        ({"CTR", "L", "R", "FL"}, ("TURN_RIGHT", "FR")),
        ({"CTR", "L", "R", "FL", "FR"}, ("STOP", None)),
        ({"FL", "L"}, ("FORWARD", "CTR")),
    ],
)
def test_direction_priority(blocked, expected):
    _, sec = _sectors()
    states = [{**s, "blocked": s["name"] in blocked} for s in sec]
    assert decide_direction(states) == expected
