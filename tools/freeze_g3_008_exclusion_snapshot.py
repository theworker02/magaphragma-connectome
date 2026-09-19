"""Freeze the immutable -008 pinned exclusion snapshot from the exposure audit.

This snapshot -- NOT directory globs -- is the -008 sampler's SOLE historical
exclusion input. It records, per package, the exact edges to avoid and why:

  HUMAN_REVIEWED         -> exact REVIEWED edges (edges that received events).
  GENERATED_NOT_PRESENTED -> exact QUEUED edges (conservative coordinate avoid),
                            but the source crop is NOT burned.

It also records, per source crop, whether the whole crop remains available
(i.e. it has no HUMAN_REVIEWED package), and the union of exact edges to avoid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "experiments/phase6e/G3_EXPOSURE_AUDIT_001.json"
OUT = REPO / "experiments/phase6e/G3_008_PINNED_EXCLUSION_SNAPSHOT_001.json"


def _edges_hash(edges: list) -> str:
    norm = sorted(tuple(map(tuple, (e[0], e[1]))) + (int(e[2]),) for e in edges)
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    audit_path = args.audit.resolve()
    audit = json.loads(audit_path.read_text())
    audit_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO)).replace("\\", "/")
        except ValueError:
            return str(p.resolve()).replace("\\", "/")

    # Per source crop: does any HUMAN_REVIEWED package consume it?
    reviewed_sources = set()
    excluded_edges_by_source: dict[str, list] = {}
    per_package = []
    for p in audit["packages"]:
        src = p["source_id"]
        if p["exposure_state"] == "HUMAN_REVIEWED":
            reviewed_sources.add(src)
            edges = p["_reviewed_edges"]
            scope = "REVIEWED_EDGES"
        else:
            edges = p["_queued_edges"]
            scope = "QUEUED_EDGES_CONSERVATIVE"
        excluded_edges_by_source.setdefault(src, [])
        excluded_edges_by_source[src].extend(edges)
        per_package.append({
            "package_id": p["package_id"],
            "source_id": src,
            "exposure_state": p["exposure_state"],
            "exclusion_scope": scope,
            "excluded_edge_count": len(edges),
            "excluded_edges_hash": _edges_hash(edges) if edges else None,
        })

    # Deduplicate per-source edges and build final avoid sets.
    source_records = []
    for src, edges in sorted(excluded_edges_by_source.items()):
        uniq = sorted({(tuple(e[0]), tuple(e[1]), int(e[2])) for e in edges})
        source_records.append({
            "source_id": src,
            "whole_crop_available": src not in reviewed_sources,
            "has_human_reviewed_package": src in reviewed_sources,
            "avoid_exact_edge_count": len(uniq),
            "avoid_edges_hash": _edges_hash([[list(a), list(b), c] for a, b, c in uniq]),
            "avoid_edges": [[list(a), list(b), c] for a, b, c in uniq],
        })

    snapshot = {
        "id": "MV-G3-008-PINNED-EXCLUSION-SNAPSHOT-001",
        "schema_version": 1,
        "status": "FROZEN_PINNED_EXCLUSION_FOR_G3_008",
        "policy": "EXPOSURE_AWARE_PROVENANCE_V1",
        "source_audit": {"path": _rel(audit_path), "sha256": audit_hash},
        "rule": {
            "HUMAN_REVIEWED": "avoid exact reviewed edges (spatial-leakage policy applied by sampler around them); source crop NOT globally burned unless sampler policy chooses to",
            "GENERATED_NOT_PRESENTED": "avoid exact queued edges only; source crop remains fully available",
            "sole_exclusion_input_for_008": True,
            "directory_globs_forbidden_as_exclusion_input": True,
        },
        "reviewed_source_ids": sorted(reviewed_sources),
        "fully_available_source_ids": sorted(r["source_id"] for r in source_records if r["whole_crop_available"]),
        "per_package": per_package,
        "per_source": source_records,
    }
    args.out.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "reviewed_source_ids": snapshot["reviewed_source_ids"],
        "fully_available_source_ids": snapshot["fully_available_source_ids"],
        "n_fully_available": len(snapshot["fully_available_source_ids"]),
        "out": _rel(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
