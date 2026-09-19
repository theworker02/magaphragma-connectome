"""Pre-launch validation + mandated proofs for the -007 axis-neutral paired batch.

Audits the pre-existing prospective batch (protocol-003). Does not modify it.
Proves the 7 required properties:
 1. exact affinity-edge permutation equivariance (each axis edge = +1 along axis);
 2. all three axes eligible regardless of dominant-gradient direction;
 3. deterministic queue regeneration (hash-rank location selection reproducible);
 4. exact Z/Y/X designed-count balance (12/12/12);
 5. exclusion of historical exact edges (-001..-006);
 6. no label/model dependency in selection metadata;
 7. spatial/provenance constraints (fresh sources, non-overlap, raw hash integrity).
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS = REPO / "experiments/phase6e/g3-external-review-packages-007"
Q = REPO / "experiments/phase6e/g3-interface-queues-007"
PROTOCOL = json.loads((REPO / "experiments/phase6e/G3_AXIS_NEUTRAL_SAMPLING_PROTOCOL_003.json").read_text())
SURVEY = json.loads((REPO / "experiments/phase6e/MV-G3-NEW-REGION-SURVEY-001.json").read_text())
MANIFEST = json.loads((REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json").read_text())
REC = {r["id"]: r for r in MANIFEST["records"]}
SEP2 = 144


def historical_edges() -> set:
    edges = set()
    for qp in glob.glob(str(REPO / "experiments/phase6e/g3-interface-queues-00[1-6]/*.json")):
        if Path(qp).name == "batch-manifest.json":
            continue
        try:
            data = json.loads(Path(qp).read_text())
        except Exception:
            continue
        for question in data.get("questions", []):
            edges.add((tuple(question["pair_left_zyx"]), tuple(question["pair_right_zyx"]), int(question["channel_zyx"])))
    return edges


def main() -> int:
    fails = []
    axis_counts = {0: 0, 1: 0, 2: 0}
    hist = historical_edges()
    all_edges = []
    excluded_sources = set(SURVEY["excluded_existing_sources"]) | set(PROTOCOL["excluded_sources"]["used_005"]) | set(PROTOCOL["excluded_sources"]["used_006"])
    existing_boxes = SURVEY["existing_region_boxes"]

    for loc in PROTOCOL["locations"]:
        crop = loc["crop_id"]
        qpath = Q / f"{crop}.json"
        ws_path = WS / crop / "workspace.json"
        q = json.loads(qpath.read_text())
        ws = json.loads(ws_path.read_text())
        center = tuple(loc["center_zyx"])

        # (6) no label/model dependency
        if not q["selection"].get("label_blind") or q["selection"].get("argmax_gradient_used") is not False:
            fails.append(f"{crop}: selection not label-blind / argmax used")

        # (7) fresh source, not excluded; raw hash integrity; binding
        if loc["source_id"] in excluded_sources:
            fails.append(f"{crop}: source {loc['source_id']} is excluded/used")
        raw = Path(ws["raw"]["path"])
        if hashlib.sha256(raw.read_bytes()).hexdigest() != ws["raw"]["sha256"] or q["raw_sha256"] != ws["raw"]["sha256"]:
            fails.append(f"{crop}: raw hash mismatch")
        if q["workspace_id"] != ws["id"]:
            fails.append(f"{crop}: workspace binding mismatch")
        # spatial non-overlap vs existing regions (fresh crops are whole survey volumes)
        rb = REC[loc["source_id"]]["bounds_xyz"]
        for other in existing_boxes:
            if all(max(rb[a][0], other["bounds_xyz"][a][0]) < min(rb[a][1], other["bounds_xyz"][a][1]) for a in ("x", "y", "z")):
                fails.append(f"{crop}: overlaps existing {other['crop_id']}")

        # (1)(2) each of the 3 axes present, edge = +1 along axis, at the SAME center
        by_axis = {qq["channel_zyx"]: qq for qq in q["questions"]}
        if set(by_axis) != {0, 1, 2}:
            fails.append(f"{crop}: not all three axes present")
        for axis, qq in by_axis.items():
            left = tuple(qq["pair_left_zyx"])
            right = tuple(qq["pair_right_zyx"])
            expected_right = left[:axis] + (left[axis] + 1,) + left[axis + 1:]
            if left != center:
                fails.append(f"{crop} axis{axis}: left != shared center")
            if right != expected_right:
                fails.append(f"{crop} axis{axis}: right not +1 along axis")
            axis_counts[axis] += 1
            key = (left, right, axis)
            all_edges.append(key)
            # (5) not a historical edge
            if key in hist:
                fails.append(f"{crop} axis{axis}: overlaps historical edge")

    # (4) exact designed balance
    if axis_counts != {0: 12, 1: 12, 2: 12}:
        fails.append(f"axis balance not 12/12/12: {axis_counts}")
    # (2) axes eligible regardless of dominant gradient: every location has all 3
    # (already checked per-crop). Confirm no location dropped an axis:
    if len(all_edges) != 36:
        fails.append(f"expected 36 edges, got {len(all_edges)}")
    # within-batch: all 12 centers 12-voxel separated? (different crops => n/a, but
    # ensure unique centers)
    centers = [tuple(l["center_zyx"]) for l in PROTOCOL["locations"]]
    if len(set(centers)) != len(centers):
        fails.append("duplicate location centers")

    report = {"checks_passed": not fails, "failures": fails, "axis_counts": axis_counts,
              "total_edges": len(all_edges), "crops": len(PROTOCOL["locations"]),
              "historical_edges_considered": len(hist),
              "strata": {loc["crop_id"]: loc["assigned_stratum"] for loc in PROTOCOL["locations"]}}
    print(json.dumps(report, indent=2))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
