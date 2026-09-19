"""Confounding / identifiability audit over ALL reviewed TRAIN regions (A-H)
plus VALIDATION, using the append-only expert event logs directly (read-only).

This does not mutate or reinterpret any frozen artifact. It characterizes the
real reviewed evidence so a new cohort-level balance/identifiability contract
can be defined and then tested.

Decision unit: the interface (interface_id). Per the frozen materializer's
contract, an interface contributes a single class only if all its member
decisions agree and are SAME/DIFFERENT; mixed => CONTRADICTORY (excluded);
UNCERTAIN/BAD_QUESTION are not class evidence. We reproduce that here so the
audit reflects usable interface-level evidence, not raw member clicks.
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

# Only the frozen-cohort TRAIN crops A-H and the three VALIDATION regions are
# in scope. -003/-004 are additional TRAIN evidence on A/B/D and are included
# as real (negative) evidence.
AXES = {0: "Z", 1: "Y", 2: "X"}


def interface_class(members: list[dict]) -> str | None:
    decisions = {m["decision"] for m in members}
    if len(decisions) != 1:
        return "CONTRADICTORY" if decisions - {"UNCERTAIN", "BAD_QUESTION"} else None
    (only,) = tuple(decisions)
    return only if only in ("SAME_PROCESS", "DIFFERENT_PROCESS") else None


def collect() -> dict:
    logs = glob.glob("experiments/phase6e/g3-external-review-packages-*/MV-G3-*/workspace.events.jsonl")
    # group events by (crop, split, interface_id) -> members
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    crop_axis_of_interface: dict[tuple, int] = {}
    for lp in logs:
        for line in open(lp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            key = (e.get("crop_id"), e.get("split"), e.get("interface_id"), lp)
            groups[key].append(e)
            crop_axis_of_interface[key] = int(e.get("channel_zyx"))
    # interface-level class per (crop, split, axis)
    matrix = collections.Counter()          # (crop, split, axis, class) -> n interfaces
    for key, members in groups.items():
        crop, split, iid, _ = key
        cls = interface_class(members)
        axis = crop_axis_of_interface[key]
        if cls in ("SAME_PROCESS", "DIFFERENT_PROCESS"):
            matrix[(crop, split, AXES[axis], cls)] += 1
        elif cls == "CONTRADICTORY":
            matrix[(crop, split, AXES[axis], "CONTRADICTORY")] += 1
    return matrix


def main() -> None:
    matrix = collect()
    crops = sorted({k[0] for k in matrix})
    axes = ["Z", "Y", "X"]
    classes = ["SAME_PROCESS", "DIFFERENT_PROCESS"]

    per_crop = {}
    for crop in crops:
        split = next(k[1] for k in matrix if k[0] == crop)
        cell = {ax: {c: matrix[(crop, split, ax, c)] for c in classes} for ax in axes}
        cell["_split"] = split
        cell["_contradictory"] = {ax: matrix[(crop, split, ax, "CONTRADICTORY")] for ax in axes}
        per_crop[crop] = cell

    # Confounding tests over TRAIN only.
    train = {c: v for c, v in per_crop.items() if v["_split"] == "G3_TARGET_TRAIN"}
    # (axis, class) coverage: which crops supply each (axis,class)?
    supply = {}
    for ax in axes:
        for c in classes:
            supply[f"{ax}:{c}"] = sorted(crop for crop in train if train[crop][ax][c] > 0)

    # crop-uniqueness: does any (axis,class) come from exactly one crop?
    unique_source = {k: v for k, v in supply.items() if len(v) == 1}
    # cross-region replication: (axis,class) present in >=2 crops
    replicated = {k: v for k, v in supply.items() if len(v) >= 2}

    # Y-DIFFERENT specifically
    y_diff_crops = supply["Y:DIFFERENT_PROCESS"]
    y_same_crops = supply["Y:SAME_PROCESS"]

    # Per-axis class<->crop collinearity: for an axis, are the crop sets that
    # supply SAME and DIFFERENT disjoint? If disjoint, class is perfectly
    # predictable from crop identity on that axis (a crop confound). We also
    # check for any single crop carrying BOTH classes on the axis (the strongest
    # within-region contrast that breaks collinearity).
    axis_identifiability = {}
    for ax in axes:
        same_crops = set(supply[f"{ax}:SAME_PROCESS"])
        diff_crops = set(supply[f"{ax}:DIFFERENT_PROCESS"])
        both_present = bool(same_crops) and bool(diff_crops)
        crops_with_both = sorted(same_crops & diff_crops)
        axis_identifiability[ax] = {
            "same_crops": sorted(same_crops),
            "different_crops": sorted(diff_crops),
            "both_classes_present_in_train": both_present,
            "class_crop_collinear": both_present and not crops_with_both,
            "crops_with_both_classes": crops_with_both,
            "replicated_same_ge2": len(same_crops) >= 2,
            "replicated_different_ge2": len(diff_crops) >= 2,
        }

    report = {
        "id": "MV-G3-IDENTIFIABILITY-AUDIT-001",
        "decision_unit": "interface_id (member-agreement required; CONTRADICTORY/UNCERTAIN/BAD excluded from class)",
        "scope": "all reviewed TRAIN A-H + VALIDATION event logs (-001..-004), read-only",
        "per_crop_interface_counts": per_crop,
        "train_axis_class_supply": supply,
        "train_axis_class_unique_source_crops": unique_source,
        "train_axis_class_replicated_across_crops": replicated,
        "Y_DIFFERENT_source_crops": y_diff_crops,
        "Y_SAME_source_crops": y_same_crops,
        "Y_class_crop_disjoint": sorted(set(y_diff_crops) & set(y_same_crops)) == [],
        "axis_identifiability": axis_identifiability,
    }
    Path("bin").mkdir(exist_ok=True)
    out = Path("experiments/phase6e/MV-G3-IDENTIFIABILITY-AUDIT-001.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
