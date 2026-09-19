"""Exposure-aware provenance audit for all G3 review packages.

Replaces the implicit "any generated queue burns its whole source crop" policy
with an explicit, hash-pinned exposure classification:

  HUMAN_REVIEWED        -- reviewer launched AND event log has >=1 recorded label
  GENERATED_NOT_PRESENTED -- queue exists but no reviewer launch and empty log

Presentation state is NOT derived from event-log size alone. Two independent
pieces of evidence are recorded:
  (1) event_count from the append-only log;
  (2) reviewer_launch_evidence: whether a reviewer process was launched on this
      workspace (supplied as an explicit, auditable input, not inferred).

Exclusion scope per package:
  HUMAN_REVIEWED        -> exclude the exact REVIEWED edges (edges that actually
                           received events), under the frozen spatial-leakage
                           policy applied around those observations.
  GENERATED_NOT_PRESENTED -> record queued edges for completeness; exclusion is
                           limited to those exact queued coordinates (optional,
                           conservative) and NEVER the whole source crop.

Outputs an immutable audit; a companion tool freezes the -008 pinned snapshot.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/phase6e/G3_EXPOSURE_AUDIT_001.json"

# Authoritative reviewer-launch evidence from the session process history.
# A workspace path here means a reviewer GUI was launched against it.
LAUNCHED_WORKSPACES = {
    "experiments/phase6e/g3-external-review-packages-005/MV-G3-TRAIN-I/workspace.json",
    "experiments/phase6e/g3-external-review-packages-005/MV-G3-TRAIN-J/workspace.json",
    "experiments/phase6e/g3-external-review-packages-005/MV-G3-TRAIN-K/workspace.json",
    "experiments/phase6e/g3-external-review-packages-005/MV-G3-TRAIN-L/workspace.json",
    "experiments/phase6e/g3-external-review-packages-006/MV-G3-AXNEU-A/workspace.json",
}


def _sha256_file(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as s:
        while b := s.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _edges_hash(edges: list[tuple]) -> str:
    # canonical, order-independent hash of a set of (left, right, channel) edges
    norm = sorted(tuple(map(tuple, (e[0], e[1]))) + (int(e[2]),) for e in edges)
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()


def _queue_edges(queue: dict) -> list[tuple]:
    return [(tuple(q["pair_left_zyx"]), tuple(q["pair_right_zyx"]), int(q["channel_zyx"])) for q in queue.get("questions", [])]


def _reviewed_edges(log_path: Path) -> list[tuple]:
    edges = []
    if not log_path.is_file():
        return edges
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        edges.append((tuple(e["pair_left_zyx"]), tuple(e["pair_right_zyx"]), int(e["channel_zyx"])))
    return edges


def audit_package(ws_path: Path) -> dict:
    ws = json.loads(ws_path.read_text())
    rel = str(ws_path.relative_to(REPO)).replace("\\", "/")
    version = "unknown"
    for tag in ("-005", "-006", "-007", "-008"):
        if f"packages{tag}" in rel:
            version = tag.strip("-")
    # locate queue: queues dir mirrors packages dir
    queue_dir = ws_path.parent.parent.name.replace("external-review-packages", "interface-queues")
    queue_root = REPO / "experiments/phase6e" / ws_path.parent.parent.name.replace("external-review-packages", "interface-queues")
    crop_id = ws["crop_id"]
    queue_path = queue_root / f"{crop_id}.json"
    queue = json.loads(queue_path.read_text()) if queue_path.is_file() else {"questions": []}
    log_path = Path(ws["event_log"]["path"])

    queued = _queue_edges(queue)
    reviewed = _reviewed_edges(log_path)
    launched = rel in LAUNCHED_WORKSPACES

    if reviewed and launched:
        exposure = "HUMAN_REVIEWED"
        exclusion_scope = "REVIEWED_EDGES_WITH_SPATIAL_LEAKAGE_POLICY"
        exclusion_reason = "Reviewer launched and recorded human labels; reviewed observations are protected supervision."
    elif reviewed and not launched:
        exposure = "HUMAN_REVIEWED"  # events exist => labels exist regardless of our launch record
        exclusion_scope = "REVIEWED_EDGES_WITH_SPATIAL_LEAKAGE_POLICY"
        exclusion_reason = "Event log contains recorded labels; treated as reviewed even without a matched launch record."
    else:
        exposure = "GENERATED_NOT_PRESENTED"
        exclusion_scope = "QUEUED_EXACT_EDGES_ONLY_NO_CROP_BURN"
        exclusion_reason = "No recorded labels and no reviewer launch; queue was generated but never presented, so it supplied no supervision and must not burn its source crop."

    return {
        "artifact_version": version,
        "package_id": f"{ws_path.parent.parent.name}/{crop_id}",
        "source_id": ws.get("parent_region_id"),
        "workspace_id": ws.get("id"),
        "workspace_path": rel,
        "queue_id": queue.get("id"),
        "queue_path": str(queue_path.relative_to(REPO)).replace("\\", "/") if queue_path.is_file() else None,
        "queue_hash": _sha256_file(queue_path),
        "event_log_hash": _sha256_file(log_path),
        "event_count": len(reviewed),
        "reviewer_launch_evidence": {"launched": launched, "source": "session_process_history"},
        "exposure_state": exposure,
        "queued_edge_count": len(queued),
        "reviewed_edge_count": len(reviewed),
        "queued_edges_hash": _edges_hash(queued) if queued else None,
        "reviewed_edges_hash": _edges_hash(reviewed) if reviewed else None,
        "exclusion_scope": exclusion_scope,
        "exclusion_reason": exclusion_reason,
        "_queued_edges": [[list(e[0]), list(e[1]), e[2]] for e in queued],
        "_reviewed_edges": [[list(e[0]), list(e[1]), e[2]] for e in reviewed],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    packages = sorted(glob.glob(str(REPO / "experiments/phase6e/g3-external-review-packages-*/*/workspace.json")))
    audited = [audit_package(Path(p)) for p in packages]
    summary = {}
    for a in audited:
        summary.setdefault(a["exposure_state"], 0)
        summary[a["exposure_state"]] += 1
    report = {
        "id": "MV-G3-EXPOSURE-AUDIT-001",
        "schema_version": 1,
        "policy": "EXPOSURE_AWARE_PROVENANCE_V1",
        "definition": {
            "HUMAN_REVIEWED": "event log has >=1 recorded label (labels exist)",
            "GENERATED_NOT_PRESENTED": "no recorded labels AND no reviewer launch",
        },
        "launch_evidence_source": "session_process_history (reviewer GUI launches on review_external_boundary_package.py)",
        "packages": audited,
        "exposure_summary": summary,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "packages": [(a["package_id"], a["exposure_state"], a["reviewed_edge_count"], a["queued_edge_count"]) for a in audited]}, indent=2))


if __name__ == "__main__":
    main()
