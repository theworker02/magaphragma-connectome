"""G3 affinity-evaluation cohort-level identifiability contract (V1).

Executable form of G3_COHORT_IDENTIFIABILITY_CONTRACT_DECISION-001. Evaluates,
per affinity axis, whether reviewed TRAIN/VALIDATION interface evidence supports
an axis-leakage-safe AND crop-confound-safe SAME/DIFFERENT evaluation.

An axis is EVALUABLE iff:
  C1 coverage: >=1 SAME and >=1 DIFFERENT interface in BOTH TRAIN and VALIDATION;
  C2 replication: within TRAIN, each class appears in >=2 distinct crops;
  C3 non-collinearity: within TRAIN, class is not perfectly predictable from
     crop identity -- i.e. the SAME-crop set and DIFFERENT-crop set are NOT
     disjoint (equivalently, >=1 crop supplies both classes on the axis).
Spatial independence (C4) is enforced by the existing G3 cohort invariant and
is checked separately by the cohort freezer, not re-derived here.

This module reads only aggregated (crop, split, axis, class) interface counts.
It never consumes raw labels beyond those counts and makes no biological claim.
"""

from __future__ import annotations

from typing import Any

AXES = ("Z", "Y", "X")
CLASSES = ("SAME_PROCESS", "DIFFERENT_PROCESS")


def evaluate_axis(axis: str, train_by_crop: dict[str, dict[str, int]], validation_by_crop: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Evaluate one axis against C1-C3.

    train_by_crop / validation_by_crop: {crop_id: {"SAME_PROCESS": n, "DIFFERENT_PROCESS": n}}
    (counts of eligible interfaces on this axis for that crop/split).
    """
    def crops_with(counts: dict[str, dict[str, int]], cls: str) -> set[str]:
        return {crop for crop, cc in counts.items() if cc.get(cls, 0) > 0}

    train_same = crops_with(train_by_crop, "SAME_PROCESS")
    train_diff = crops_with(train_by_crop, "DIFFERENT_PROCESS")
    val_same = crops_with(validation_by_crop, "SAME_PROCESS")
    val_diff = crops_with(validation_by_crop, "DIFFERENT_PROCESS")

    c1 = bool(train_same) and bool(train_diff) and bool(val_same) and bool(val_diff)
    c2 = len(train_same) >= 2 and len(train_diff) >= 2
    both = sorted(train_same & train_diff)
    c3 = bool(both)  # non-collinear iff >=1 crop supplies both classes

    return {
        "axis": axis,
        "train_same_crops": sorted(train_same),
        "train_different_crops": sorted(train_diff),
        "validation_same_crops": sorted(val_same),
        "validation_different_crops": sorted(val_diff),
        "C1_coverage_both_splits": c1,
        "C2_train_replication_ge2": c2,
        "C3_non_collinear_crops_with_both": both,
        "C3_non_collinear": c3,
        "evaluable": bool(c1 and c2 and c3),
        "failure_reasons": [
            name for name, ok in (
                ("C1_COVERAGE", c1), ("C2_REPLICATION", c2), ("C3_NON_COLLINEARITY", c3)
            ) if not ok
        ],
    }


def evaluate_cohort(per_axis_train: dict[str, dict[str, dict[str, int]]], per_axis_validation: dict[str, dict[str, dict[str, int]]]) -> dict[str, Any]:
    """Evaluate all axes. per_axis_* : {axis: {crop: {class: count}}}."""
    axes = {}
    for axis in AXES:
        axes[axis] = evaluate_axis(axis, per_axis_train.get(axis, {}), per_axis_validation.get(axis, {}))
    evaluable = [a for a, r in axes.items() if r["evaluable"]]
    return {
        "contract": "G3_AFFINITY_EVAL_COHORT_LEVEL_IDENTIFIABILITY_CONTRACT_V1",
        "axes": axes,
        "evaluable_axes": evaluable,
        "cohort_valid": bool(evaluable),
    }
