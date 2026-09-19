"""Specimen-safe Phase 5 campaign planning from an already ingested real volume.

This intentionally does not run segmentation: FFN execution is owned by the
registered reconstruction backend.  It creates the durable denominator and
work queue to which that backend must attach its real runs.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .io import sha256_file, write_json_atomic


def _now() -> str:
    """Return a UTC timestamp for durable campaign-planning receipts."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def initialize(ingestion_path: Path, output: Path, phase4_build: Path) -> dict:
    """Initialize an auditable work queue without executing or claiming segmentation."""
    ingestion = json.loads(ingestion_path.read_text(encoding="utf-8"))
    bounds = ingestion["source_bounds_xyz"]
    # Source attribution exists, but this public DVID crop has no verified specimen
    # equivalence to the CATMAID source specimen.  Unknown is a hard boundary.
    domain = {
        "id": "MV-PHASE5-DOMAIN-001", "specimen_id": "SPECIMEN_UNKNOWN",
        "volume_id": ingestion["volume_id"], "dataset_id": ingestion["dataset_id"],
        "source_bounds_xyz": bounds, "voxel_size_nm_xyz": [8.0, 8.0, 8.0],
        "axis_order": "XYZ", "array_order": "ZYX", "coordinate_frame": "MV-FRAME-DVID-WASP5-001",
        "source_raw_sha256": ingestion["raw_sha256"], "raw_data_immutable": True,
        "source_to_catmaid_transform": {"status": "UNKNOWN", "effect": "CATMAID source neurons cannot be anchored here"},
        "eligibility": "LOCAL_PROCESSING_ONLY", "excluded_regions": [],
        "known_missing_slabs": [], "note": "Defined denominator is this bounded real-EM crop, not a whole specimen or whole brain.",
    }
    output.mkdir(parents=True, exist_ok=True)
    # Four cores per XY plane × two Z cores, with documented source-bound read margins.
    chunks = []
    serial = 1
    for z0 in range(bounds["z"][0], bounds["z"][1], 32):
        for y0 in range(bounds["y"][0], bounds["y"][1], 256):
            for x0 in range(bounds["x"][0], bounds["x"][1], 256):
                core = {"x": [x0, min(x0 + 256, bounds["x"][1])], "y": [y0, min(y0 + 256, bounds["y"][1])], "z": [z0, min(z0 + 32, bounds["z"][1])]}
                read = {axis: [max(bounds[axis][0], core[axis][0] - (16 if axis != "z" else 8)), min(bounds[axis][1], core[axis][1] + (16 if axis != "z" else 8))] for axis in ("x", "y", "z")}
                chunks.append({"id": f"MV-CHUNK-{serial:08d}", "domain_id": domain["id"], "specimen_id": "SPECIMEN_UNKNOWN", "volume_id": domain["volume_id"], "core_bounds_xyz": core, "read_bounds_xyz": read, "overlap_margin_voxels_xyz": [16,16,8], "processing_state": "NOT_STARTED", "segmentation_state": "NOT_STARTED", "skeleton_state": "NOT_STARTED", "synapse_state": "NOT_STARTED", "review_state": "UNREVIEWED", "retry_count": 0})
                serial += 1
    source_manifest = json.loads((phase4_build / "manifest.json").read_text(encoding="utf-8"))
    baseline = {"id": "MV-PRE-PHASE5-BASELINE", "generated_at": _now(), "release_eligible": False,
                "phase4_build_manifest_sha256": sha256_file(phase4_build / "manifest.json"),
                "catmaid_source_artifact_sha256": source_manifest["source_artifact_sha256"],
                "phase4_counts": source_manifest["counts"], "domain": domain,
                "ffn": {"status": "AWAITING_REGISTERED_BACKEND_RUN", "no_ffn_run_is_claimed_by_this_initializer": True}}
    frontier = [{"id": f"MV-FRONTIER-{i:08d}", "kind": "DVID_NATIVE_TRACE", "chunk_id": c["id"], "status": "QUEUED", "reason": "PHASE5_FIRST_PASS_UNREGISTERED", "priority_components": {"coverage_impact": 1.0, "source_evidence": 0.0, "review_cost": 0.0}} for i,c in enumerate(chunks,1)]
    write_json_atomic(output / "domain.json", domain)
    write_json_atomic(output / "chunks.json", {"chunks": chunks})
    write_json_atomic(output / "frontier.json", {"tasks": frontier})
    write_json_atomic(output / "MV-PRE-PHASE5-BASELINE.json", baseline)
    return {"domain": domain["id"], "chunks": len(chunks), "frontier_tasks": len(frontier), "baseline": baseline["id"]}
