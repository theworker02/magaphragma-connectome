"""Immutable exposure audit for prospective batches -005, -006, -007.

Classifies every source/edge into exposure states using ONLY immutable
evidence: event logs, queue/workspace provenance, artifact presence, and exact
edge coordinates. Biological labels are not inspected beyond confirming that an
event line exists (an event's existence is itself the exposure signal).

Exposure states:
  HUMAN_REVIEWED                    - >=1 append-only event references the edge
  PRESENTED_TO_REVIEWER_BUT_UNLABELED - crop's queue was opened in a reviewer
                                        session but the edge has no event
  GENERATED_NOT_PRESENTED           - queue/edge generated, never opened in a
                                        reviewer, no events
  NEVER_SELECTED                    - not present in any prospective queue

Determination of "presented in a reviewer": we treat a crop as presented iff
its workspace event log is non-empty (>=1 event) OR an explicit reviewer-session
record exists. Absent a session ledger, a completely empty event log with no
launched reviewer is classified GENERATED_NOT_PRESENTED. This is recorded as an
assumption in the output so it is auditable.

Writes G3_PROSPECTIVE_EXPOSURE_AUDIT_001.json plus a pinned, hashed exclusion
snapshot G3_008_EXCLUSION_SNAPSHOT_001.json.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PHASE = REPO / "experiments/phase6e"
AUDIT_OUT = PHASE / "G3_PROSPECTIVE_EXPOSURE_AUDIT_001.json"
SNAPSHOT_OUT = PHASE / "G3_008_EXCLUSION_SNAPSHOT_001.json"

# Spatial leakage radius for HUMAN_REVIEWED edges. The frozen G3 contract uses
# strict spatial disjointness at the region level; for edge-level prospective
# reuse we honor a conservative neighborhood radius (voxels) around reviewed
# edges. Whole-crop exclusion is applied ONLY for human-reviewed crops, matching
# the historical review-admission behavior; unlabeled/generated crops are not
# whole-crop excluded.
HUMAN_EDGE_LEAKAGE_RADIUS = 12
PRESENTED_EDGE_LEAKAGE_RADIUS = 12

TIERS = ["005", "006", "007"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as s:
        while b := s.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _events(log_path: Path) -> list[dict]:
    if not log_path.is_file():
        return []
    out = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _edge_key(e: dict) -> tuple:
    return (tuple(e["pair_left_zyx"]), tuple(e["pair_right_zyx"]), int(e["channel_zyx"]))


def audit_tier(tier: str) -> list[dict]:
    records = []
    pkg_root = PHASE / f"g3-external-review-packages-{tier}"
    q_root = PHASE / f"g3-interface-queues-{tier}"
    if not pkg_root.is_dir():
        return records
    for crop_dir in sorted(p for p in pkg_root.iterdir() if p.is_dir()):
        crop_id = crop_dir.name
        ws_path = crop_dir / "workspace.json"
        ws = json.loads(ws_path.read_text())
        source_id = ws.get("parent_region_id")
        log = crop_dir / "workspace.events.jsonl"
        events = _events(log)
        reviewed_edges = {_edge_key(e) for e in events}
        queue_path = q_root / f"{crop_id}.json"
        queue = json.loads(queue_path.read_text()) if queue_path.is_file() else {"questions": []}
        queue_edges = {(tuple(q["pair_left_zyx"]), tuple(q["pair_right_zyx"]), int(q["channel_zyx"])) for q in queue.get("questions", [])}

        crop_presented = len(events) > 0  # conservative: empty log + no session => not presented
        for edge in sorted(queue_edges):
            if edge in reviewed_edges:
                state = "HUMAN_REVIEWED"
                radius = HUMAN_EDGE_LEAKAGE_RADIUS
            elif crop_presented:
                state = "PRESENTED_TO_REVIEWER_BUT_UNLABELED"
                radius = PRESENTED_EDGE_LEAKAGE_RADIUS
            else:
                state = "GENERATED_NOT_PRESENTED"
                radius = 0  # exact-edge exclusion only, not neighborhood
            records.append({
                "tier": tier,
                "crop_id": crop_id,
                "source_id": source_id,
                "workspace_id": ws.get("id"),
                "pair_left_zyx": list(edge[0]),
                "pair_right_zyx": list(edge[1]),
                "channel_zyx": edge[2],
                "exposure_state": state,
                "leakage_radius_voxels": radius,
                "originating_artifact": str(queue_path.resolve()),
                "event_log": str(log.resolve()),
                "crop_presented": crop_presented,
            })
    return records


def main() -> None:
    all_records = []
    for tier in TIERS:
        all_records.extend(audit_tier(tier))

    # Per-source rollup
    by_source: dict[str, dict] = {}
    for r in all_records:
        s = by_source.setdefault(r["source_id"], {"source_id": r["source_id"], "states": {}, "crops": set(), "human_reviewed_crop": False})
        s["states"][r["exposure_state"]] = s["states"].get(r["exposure_state"], 0) + 1
        s["crops"].add(r["crop_id"])
        if r["exposure_state"] == "HUMAN_REVIEWED":
            s["human_reviewed_crop"] = True
    for s in by_source.values():
        s["crops"] = sorted(s["crops"])

    audit = {
        "id": "MV-G3-PROSPECTIVE-EXPOSURE-AUDIT-001",
        "schema_version": 1,
        "assumptions": {
            "presented_definition": "A crop is PRESENTED iff its append-only event log is non-empty. No separate reviewer-session ledger exists; an empty log with no launched reviewer is GENERATED_NOT_PRESENTED.",
            "human_reviewed_crop_whole_crop_excluded": True,
            "presented_unlabeled_whole_crop_excluded": False,
            "generated_not_presented_whole_crop_excluded": False,
        },
        "exposure_states_definition": {
            "HUMAN_REVIEWED": "edge referenced by >=1 append-only event",
            "PRESENTED_TO_REVIEWER_BUT_UNLABELED": "crop presented (non-empty log) but this edge unlabeled",
            "GENERATED_NOT_PRESENTED": "edge generated in a queue, crop never presented",
            "NEVER_SELECTED": "not present in any prospective queue",
        },
        "edge_records": all_records,
        "per_source_rollup": list(by_source.values()),
        "counts": {
            "total_edges": len(all_records),
            "HUMAN_REVIEWED": sum(r["exposure_state"] == "HUMAN_REVIEWED" for r in all_records),
            "PRESENTED_TO_REVIEWER_BUT_UNLABELED": sum(r["exposure_state"] == "PRESENTED_TO_REVIEWER_BUT_UNLABELED" for r in all_records),
            "GENERATED_NOT_PRESENTED": sum(r["exposure_state"] == "GENERATED_NOT_PRESENTED" for r in all_records),
        },
    }
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Pinned exclusion snapshot for -008: exact edges + scopes + reasons.
    exclusions = []
    human_reviewed_sources = sorted({r["source_id"] for r in all_records if r["exposure_state"] == "HUMAN_REVIEWED"})
    for r in all_records:
        if r["exposure_state"] == "GENERATED_NOT_PRESENTED":
            # exact-edge conservative exclusion only; crop remains eligible
            scope = "EXACT_EDGE"
        else:
            scope = f"EDGE_PLUS_RADIUS_{r['leakage_radius_voxels']}"
        exclusions.append({
            "source_id": r["source_id"],
            "pair_left_zyx": r["pair_left_zyx"],
            "pair_right_zyx": r["pair_right_zyx"],
            "channel_zyx": r["channel_zyx"],
            "spatial_exclusion_radius_voxels": r["leakage_radius_voxels"],
            "exclusion_scope": scope,
            "exposure_state": r["exposure_state"],
            "originating_artifact": r["originating_artifact"],
            "reason": {
                "HUMAN_REVIEWED": "reviewed evidence; exact edge + leakage neighborhood excluded; whole reviewed crop excluded",
                "PRESENTED_TO_REVIEWER_BUT_UNLABELED": "presented but unlabeled; exact edge + neighborhood excluded; crop NOT whole-excluded",
                "GENERATED_NOT_PRESENTED": "generated by superseded sampler, never presented; exact edge excluded conservatively; crop NOT consumed",
            }[r["exposure_state"]],
        })
    snapshot = {
        "id": "MV-G3-008-EXCLUSION-SNAPSHOT-001",
        "schema_version": 1,
        "derived_from_audit": {"path": str(AUDIT_OUT.resolve()), "sha256": _sha256(AUDIT_OUT)},
        "whole_crop_excluded_sources": {
            "reason": "human-reviewed crops are whole-crop excluded per historical review-admission behavior",
            "sources": human_reviewed_sources,
        },
        "exact_edge_exclusions": exclusions,
        "policy": {
            "HUMAN_REVIEWED": {"whole_crop_excluded": True, "edge_radius": HUMAN_EDGE_LEAKAGE_RADIUS},
            "PRESENTED_TO_REVIEWER_BUT_UNLABELED": {"whole_crop_excluded": False, "edge_radius": PRESENTED_EDGE_LEAKAGE_RADIUS},
            "GENERATED_NOT_PRESENTED": {"whole_crop_excluded": False, "edge_radius": 0},
        },
    }
    SNAPSHOT_OUT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshot_hash = _sha256(SNAPSHOT_OUT)

    print(json.dumps({
        "audit": str(AUDIT_OUT.resolve()),
        "snapshot": str(SNAPSHOT_OUT.resolve()),
        "snapshot_sha256": snapshot_hash,
        "counts": audit["counts"],
        "human_reviewed_sources": human_reviewed_sources,
        "per_source": [{"source": s["source_id"], "states": s["states"], "human_crop": s["human_reviewed_crop"]} for s in by_source.values()],
    }, indent=2))


if __name__ == "__main__":
    main()
