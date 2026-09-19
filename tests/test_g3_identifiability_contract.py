"""Tests for the cohort-level identifiability contract (C1/C2/C3).

Includes a regression fixture reproducing the real A-H Y-axis situation:
SAME and DIFFERENT come from disjoint crop sets => C3 must FAIL.
"""
from mvconnectome.g3_identifiability_contract import evaluate_axis, evaluate_cohort


def _counts(**crops):
    # crops: crop=(same,diff)
    return {c: {"SAME_PROCESS": s, "DIFFERENT_PROCESS": d} for c, (s, d) in crops.items()}


def test_real_Y_axis_is_crop_collinear_and_fails_c3():
    # A/B/D/F/G/H supply Y SAME only; C/E supply Y DIFFERENT only (disjoint).
    train = _counts(A=(13, 0), B=(11, 0), D=(9, 0), F=(4, 0), G=(5, 0), H=(2, 0), C=(0, 4), E=(0, 6))
    val = _counts(V3=(3, 3))  # validation happens to have both
    r = evaluate_axis("Y", train, val)
    assert r["C1_coverage_both_splits"] is True
    assert r["C2_train_replication_ge2"] is True
    assert r["C3_non_collinear"] is False
    assert r["evaluable"] is False
    assert "C3_NON_COLLINEARITY" in r["failure_reasons"]


def test_axis_passes_when_a_crop_has_both_classes():
    # A single crop with both classes breaks collinearity.
    train = _counts(A=(5, 3), B=(4, 0), C=(0, 2))
    val = _counts(V1=(2, 2))
    r = evaluate_axis("Y", train, val)
    assert r["C3_non_collinear"] is True
    assert r["C3_non_collinear_crops_with_both"] == ["A"]
    assert r["evaluable"] is True


def test_c1_fails_without_validation_class():
    train = _counts(A=(5, 3), B=(2, 2))
    val = _counts(V1=(3, 0))  # no DIFFERENT in validation
    r = evaluate_axis("Y", train, val)
    assert r["C1_coverage_both_splits"] is False
    assert r["evaluable"] is False


def test_c2_fails_with_single_crop_class():
    # DIFFERENT only from one crop.
    train = _counts(A=(5, 0), B=(4, 0), E=(0, 3))
    val = _counts(V1=(2, 2))
    r = evaluate_axis("X", train, val)
    assert r["C2_train_replication_ge2"] is False
    assert r["evaluable"] is False


def test_cohort_invalid_when_no_axis_evaluable():
    per_axis_train = {
        "Z": _counts(A=(0, 3), B=(0, 3)),               # no SAME anywhere
        "Y": _counts(A=(13, 0), B=(11, 0), C=(0, 4), E=(0, 6)),  # collinear
        "X": _counts(A=(1, 0), E=(0, 2)),               # unique + collinear
    }
    per_axis_val = {
        "Z": _counts(V1=(0, 2)),
        "Y": _counts(V3=(3, 3)),
        "X": _counts(V2=(4, 0)),
    }
    result = evaluate_cohort(per_axis_train, per_axis_val)
    assert result["cohort_valid"] is False
    assert result["evaluable_axes"] == []


def test_cohort_valid_with_engineered_within_region_contrast():
    per_axis_train = {"Y": _counts(A=(5, 2), B=(3, 0), C=(0, 4))}
    per_axis_val = {"Y": _counts(V1=(2, 2))}
    result = evaluate_cohort(per_axis_train, per_axis_val)
    assert result["cohort_valid"] is True
    assert result["evaluable_axes"] == ["Y"]
