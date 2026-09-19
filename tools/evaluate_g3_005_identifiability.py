"""Evaluate the identifiability contract (unchanged) on the combined evidence:
existing TRAIN A-H + new TRAIN I-L (-005) + VALIDATION. Interface-level.

Runs src/mvconnectome/g3_identifiability_contract.evaluate_cohort exactly as
written. No thresholds are altered here.
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

from mvconnectome.g3_identifiability_contract import AXES, evaluate_cohort

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/phase6e/MV-G3-IDENTIFIABILITY-AUDIT-002.json"


def interface_class(members):
    decisions = {m["decision"] for m in members}
    if len(decisions) != 1:
        return "CONTRADICTORY" if decisions - {"UNCERTAIN", "BAD_QUESTION"} else None
    (only,) = tuple(decisions)
    return only if only in ("SAME_PROCESS", "DIFFERENT_PROCESS") else None


def collect():
    logs = glob.glob(str(REPO / "experiments/phase6e/g3-external-review-packages-*/MV-G3-*/workspace.events.jsonl"))
    groups = collections.defaultdict(list)
    axis_of = {}
    for lp in logs:
        for line in open(lp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            key = (e.get("crop_id"), e.get("split"), e.get("interface_id"), lp)
            groups[key].append(e)
            axis_of[key] = e.get("channel_name")
    # per (split, axis) -> {crop: {class: count}}
    train = {ax: collections.defaultdict(lambda: {"SAME_PROCESS": 0, "DIFFERENT_PROCESS": 0}) for ax in AXES}
    val = {ax: collections.defaultdict(lambda: {"SAME_PROCESS": 0, "DIFFERENT_PROCESS": 0}) for ax in AXES}
    for key, members in groups.items():
        crop, split, iid, _ = key
        cls = interface_class(members)
        ax = axis_of[key]
        if cls not in ("SAME_PROCESS", "DIFFERENT_PROCESS"):
            continue
        target = train if split == "G3_TARGET_TRAIN" else val if split == "G3_TARGET_VALIDATION" else None
        if target is None:
            continue
        target[ax][crop][cls] += 1
    # to plain dicts
    def plain(d):
        return {ax: {crop: dict(cc) for crop, cc in d[ax].items()} for ax in AXES}
    return plain(train), plain(val)


def main():
    per_axis_train, per_axis_val = collect()
    result = evaluate_cohort(per_axis_train, per_axis_val)
    report = {
        "id": "MV-G3-IDENTIFIABILITY-AUDIT-002",
        "scope": "TRAIN A-H + I-L (-005) + VALIDATION; interface-level; contract unchanged",
        "per_axis_train_crop_class_counts": per_axis_train,
        "per_axis_validation_crop_class_counts": per_axis_val,
        "contract_result": result,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cohort_valid": result["cohort_valid"], "evaluable_axes": result["evaluable_axes"],
                      "axes": {a: {"C1": r["C1_coverage_both_splits"], "C2": r["C2_train_replication_ge2"],
                                    "C3": r["C3_non_collinear"], "evaluable": r["evaluable"],
                                    "crops_with_both": r["C3_non_collinear_crops_with_both"]}
                               for a, r in result["axes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
