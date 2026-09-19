"""Deterministic, label-blind selection of NEW G3 TRAIN crops from the eligible
survey population. The rule is fixed here BEFORE any expert label is generated.

Selection rule SELECT_V1 (deterministic, label-blind):
  1. Start from MV-G3-NEW-REGION-SURVEY-001 eligible candidates.
  2. Rank by a fixed key that uses only geometry:
       primary: DESC min_gap_to_any_existing  (maximize spatial independence)
       tie-break: ASC candidate_source_id      (stable, deterministic)
  3. Greedily accept candidates, enforcing SPATIAL DIVERSITY: a newly accepted
     crop must occupy a distinct (y-band) from already-accepted crops so the new
     set is not spatially clustered, AND must be mutually non-overlapping with
     accepted crops (guaranteed by survey, re-checked).
  4. Stop at TARGET_N = 4.
No SAME/DIFFERENT label, prediction, learned affinity, or A-H class composition
is consulted. min_gap and y-band are raw coordinate geometry only.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SURVEY = REPO / "experiments/phase6e/MV-G3-NEW-REGION-SURVEY-001.json"
OUT = REPO / "experiments/phase6e/G3_NEW_REGION_SELECTION_PROTOCOL_001.json"
TARGET_N = 4


def _overlap(a: dict, b: dict) -> bool:
    return all(max(a[x][0], b[x][0]) < min(a[x][1], b[x][1]) for x in ("x", "y", "z"))


def main() -> None:
    survey = json.loads(SURVEY.read_text())
    pop = {e["candidate_source_id"]: e for e in survey["surveyed_population"]}
    eligible = [pop[sid] for sid in survey["eligible_candidates"]]

    ranked = sorted(eligible, key=lambda e: (-int(e["min_gap_to_any_existing"]), e["candidate_source_id"]))

    accepted: list[dict] = []
    used_ybands: set[tuple] = set()
    considered = []
    for e in ranked:
        yb = tuple(e["bounds_xyz"]["y"])
        reason = None
        if yb in used_ybands:
            reason = "SKIP_DUPLICATE_Y_BAND_FOR_SPATIAL_DIVERSITY"
        elif any(_overlap(e["bounds_xyz"], a["bounds_xyz"]) for a in accepted):
            reason = "SKIP_OVERLAP_WITH_ACCEPTED"
        considered.append({"candidate": e["candidate_source_id"], "min_gap": e["min_gap_to_any_existing"], "y_band": list(yb), "accepted": reason is None, "skip_reason": reason})
        if reason is None:
            accepted.append(e)
            used_ybands.add(yb)
        if len(accepted) >= TARGET_N:
            break

    if len(accepted) < TARGET_N:
        raise SystemExit(f"Only {len(accepted)} spatially-diverse eligible crops available; need {TARGET_N}")

    new_crop_ids = {e["candidate_source_id"]: f"MV-G3-TRAIN-{chr(ord('I') + i)}" for i, e in enumerate(accepted)}
    protocol = {
        "id": "MV-G3-NEW-REGION-SELECTION-PROTOCOL-001",
        "schema_version": 1,
        "status": "PREDECLARED_SELECTION_FIXED_BEFORE_LABELS",
        "selection_rule": "SELECT_V1",
        "rule_description": "Rank eligible survey candidates by DESC min_gap_to_any_existing, tie-break ASC source_id; greedily accept enforcing distinct y-band and mutual non-overlap; stop at 4.",
        "label_blind": True,
        "forbidden_inputs": ["SAME/DIFFERENT labels", "class predictions", "learned affinities", "A-H class composition"],
        "survey": {"path": str(SURVEY.resolve())},
        "target_n": TARGET_N,
        "consideration_trace": considered,
        "selected": [
            {
                "new_crop_id": new_crop_ids[e["candidate_source_id"]],
                "source_id": e["candidate_source_id"],
                "bounds_xyz": e["bounds_xyz"],
                "shape_zyx": e["shape_zyx"],
                "raw_sha256": e["raw_sha256"],
                "min_gap_to_any_existing": e["min_gap_to_any_existing"],
            }
            for e in accepted
        ],
        "new_split": "G3_TARGET_TRAIN",
    }
    OUT.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected": [(s["new_crop_id"], s["source_id"], s["bounds_xyz"]) for s in protocol["selected"]], "out": str(OUT.resolve())}, indent=2))


if __name__ == "__main__":
    main()
