"""Finalize presentation-repair smoke after all 12 human decisions.

Unblinds the sealed axis map, evaluates the predeclared verdict rule, and
updates G3_PRESENTATION_REPAIR_SMOKE_001.json. Fail-closed if incomplete.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "experiments/phase6e/G3_PRESENTATION_REPAIR_SMOKE_001.json"
SEALED = REPO / "experiments/phase6e/G3_PRESENTATION_REPAIR_SMOKE_AXIS_MAP_SEALED_001.json"
WS_ROOT = REPO / "experiments/phase6e/g3-presentation-repair-smoke-packages"
VALID = {"SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"}
PATTERN = {"Z": "DIFFERENT_PROCESS", "Y": "SAME_PROCESS", "X": "SAME_PROCESS"}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def main() -> int:
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    if smoke.get("verdict"):
        raise SystemExit("Smoke already has a verdict; refusing to overwrite")

    by_opaque = {e["opaque_decision_id"]: e for e in sealed["entries"]}
    decisions = {}
    issues = []
    for loc in smoke["per_location"]:
        crop = loc["crop_id"]
        log = WS_ROOT / crop / "workspace.events.jsonl"
        if not log.exists() or not log.read_text(encoding="utf-8").strip():
            issues.append(f"{crop}: empty event log")
            continue
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            oid = ev.get("question_reference")
            if oid not in by_opaque:
                issues.append(f"{crop}: unknown opaque id {oid}")
                continue
            if ev.get("decision") not in VALID:
                issues.append(f"{crop}: bad decision {ev.get('decision')}")
                continue
            meta = by_opaque[oid]
            # Integrity: event geometry must match sealed map
            if list(ev["pair_left_zyx"]) != meta["pair_left_zyx"] or int(ev["channel_zyx"]) != meta["channel_zyx"]:
                issues.append(f"{oid}: event/sealed geometry mismatch")
            if oid in decisions:
                issues.append(f"{oid}: duplicate decision")
            decisions[oid] = {
                "opaque_decision_id": oid,
                "decision": ev["decision"],
                "event_id": ev["id"],
                "crop_id": crop,
                "source_id": meta["source_id"],
                "center_zyx": meta["center_zyx"],
                "axis_name": meta["axis_name"],
                "channel_zyx": meta["channel_zyx"],
                "pair_left_zyx": meta["pair_left_zyx"],
                "pair_right_zyx": meta["pair_right_zyx"],
                "reviewer": ev.get("reviewer"),
                "timestamp": ev.get("timestamp"),
            }

    if len(decisions) != 12 or issues:
        print(json.dumps({"complete": False, "n": len(decisions), "issues": issues}, indent=2))
        return 1

    # Unblind triplets per location
    triplets = []
    pathological = 0
    evaluable = 0
    for loc in smoke["per_location"]:
        crop = loc["crop_id"]
        labels = {}
        excluded = False
        for oid, meta in by_opaque.items():
            if meta["crop_id"] != crop:
                continue
            d = decisions[oid]["decision"]
            if d in smoke["predeclared_interpretation"]["exclusions_from_pattern_test"]:
                excluded = True
            labels[meta["axis_name"]] = d
        entry = {
            "crop_id": crop,
            "source_id": loc["source_id"],
            "center_zyx": loc["center_zyx"],
            "labels_zyx": labels,
            "excluded_from_pattern_test": excluded,
            "matches_pathological_pattern": (not excluded) and labels == PATTERN,
        }
        if not excluded:
            evaluable += 1
            if entry["matches_pathological_pattern"]:
                pathological += 1
        triplets.append(entry)

    if evaluable == 0:
        verdict = "PRESENTATION_FIX_INSUFFICIENT"
        detail = "No evaluable triplets (all UNCERTAIN/BAD); treat as insufficient evidence for causal support."
    elif pathological == evaluable:
        verdict = "PRESENTATION_FIX_INSUFFICIENT"
        detail = f"All {evaluable} evaluable triplets match Z=DIFF,Y=SAME,X=SAME."
    else:
        verdict = "PRESENTATION_DEFECT_CAUSALLY_SUPPORTED"
        detail = f"{pathological}/{evaluable} evaluable triplets match pathological pattern; pattern broken."

    smoke["status"] = "COMPLETE_UNBLINDED"
    smoke["completed_at"] = _now()
    smoke["results"] = {
        "n_decisions": 12,
        "triplets": triplets,
        "evaluable_triplets": evaluable,
        "pathological_triplets": pathological,
        "decisions": sorted(decisions.values(), key=lambda d: d["opaque_decision_id"]),
    }
    smoke["verdict"] = verdict
    smoke["verdict_detail"] = detail
    smoke["blinding"]["unblinded_at"] = _now()
    smoke["do_not_train"] = True
    smoke["smoke_labels_are_not_production_gt"] = True

    sealed["status"] = "UNBLINDED"
    sealed["unblinded_at"] = _now()
    SEALED.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SMOKE.write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "detail": detail, "triplets": triplets}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
