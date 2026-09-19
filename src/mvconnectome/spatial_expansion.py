"""Freeze spatially diverse DVID evidence cohorts for target adaptation.

This module deliberately separates region selection from G1 diagnostic output.
Regions are frozen first, then a fixed G1 checkpoint may rank *within* those
regions for review.  The protected regression region is never eligible here.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import sha256_file, write_json_atomic

_ROLES = {"G2_TARGET_TRAIN", "G2_TARGET_VALIDATION"}
_PROTECTED = "MV-GTVOL-000004"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _box(record: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    bounds = record["bounds_xyz"]
    return tuple(tuple(int(v) for v in bounds[axis]) for axis in ("x", "y", "z"))  # type: ignore[return-value]


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(max(a[0], b[0]) < min(a[1], b[1]) for a, b in zip(_box(left), _box(right), strict=True))


def freeze_spatial_expansion_cohort(*, survey_manifest_path: Path, plan_path: Path, output_path: Path) -> dict[str, Any]:
    """Write one immutable, split-safe G2 cohort manifest.

    `plan.selected_regions` is a list of source survey identifiers and roles.
    Selection must not be based on a model prediction; model diagnostics are
    explicitly requested only after this function has frozen the raw inputs.
    """
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen cohort: {output_path}")
    survey = json.loads(survey_manifest_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source = {entry["id"]: entry for entry in survey.get("records", [])}
    selected = plan.get("selected_regions", [])
    if len(selected) < 2:
        raise ValueError("A G2 cohort requires at least two spatial regions")
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for item in selected:
        identifier, role = item.get("source_id"), item.get("role")
        if not isinstance(identifier, str) or identifier in seen:
            raise ValueError("Selected source IDs must be unique")
        if role not in _ROLES:
            raise ValueError(f"{identifier}: invalid G2 role")
        original = source.get(identifier)
        if original is None:
            raise ValueError(f"Unknown source survey region: {identifier}")
        raw = Path(original["raw_path"])
        if not raw.is_file() or sha256_file(raw) != original["raw_sha256"]:
            raise ValueError(f"{identifier}: immutable raw survey file/hash mismatch")
        if _PROTECTED in json.dumps(original, sort_keys=True):
            raise ValueError("Protected regression volume cannot enter G2 supervision")
        seen.add(identifier)
        records.append({
            "id": item.get("id", f"MV-G2-{role.split('_')[-1]}-{len(records) + 1:03d}"),
            "role": role,
            "source_id": identifier,
            "bounds_xyz": original["bounds_xyz"],
            "raw_path": str(raw.resolve()),
            "raw_sha256": original["raw_sha256"],
            "shape_zyx": original["shape_zyx"],
            "dtype": original["dtype"],
            "source_url": original["source_url"],
            "voxel_size_nm_xyz": survey["source"]["voxel_size_nm_xyz"],
            "status": "FROZEN_PENDING_G1_DIAGNOSTIC_AND_EXPERT_REVIEW",
        })
    if not any(record["role"] == "G2_TARGET_TRAIN" for record in records):
        raise ValueError("G2 requires at least one target TRAIN region")
    if not any(record["role"] == "G2_TARGET_VALIDATION" for record in records):
        raise ValueError("G2 requires at least one target VALIDATION region")
    for index, left in enumerate(records):
        for right in records[index + 1:]:
            if _overlap(left, right):
                raise ValueError(f"Spatially overlapping G2 regions: {left['id']} / {right['id']}")
    existing = plan.get("existing_reviewed_regions", [])
    for prior in existing:
        if _PROTECTED in json.dumps(prior, sort_keys=True):
            raise ValueError("Protected regression volume cannot be registered as reviewed evidence")
        for record in records:
            if _overlap(prior, record):
                raise ValueError(f"New G2 region {record['id']} overlaps frozen reviewed region {prior.get('id')}")
    reserved = plan.get("reserved_evaluation_regions", [])
    for held_out in reserved:
        if _PROTECTED in json.dumps(held_out, sort_keys=True):
            raise ValueError("Protected regression volume must not be encoded as a candidate supervision region")
        for record in records:
            if _overlap(held_out, record):
                raise ValueError(f"New G2 region {record['id']} overlaps reserved evaluation region {held_out.get('id')}")
    train = [record["id"] for record in records if record["role"] == "G2_TARGET_TRAIN"]
    loro = [
        {"held_out_region": identifier, "train_with": [other for other in train if other != identifier],
         "criterion": "reviewed SAME and DIFFERENT decisions in held-out region are excluded from G2 fitting and assessed separately"}
        for identifier in train
    ]
    value = {
        "schema_version": 1,
        "id": plan.get("id", "MV-G2-SPATIAL-EXPANSION-COHORT-001"),
        "created_at": _now(),
        "status": "FROZEN_BEFORE_G1_DIAGNOSTIC_REVIEW",
        "source_survey": {"path": str(survey_manifest_path.resolve()), "sha256": sha256_file(survey_manifest_path)},
        "selection": {
            "method": "PRE_G1_SPATIALLY_STRATIFIED_REGION_SELECTION_V1",
            "prohibited_inputs": ["G1 prediction", "G1 reconstruction quality", "MV-GTVOL-000004", "review outcomes"],
            "post_freeze_diagnostic": "Use frozen G1 step2280 output only to rank candidate merge-confident/raw-EM-boundary contradictions and continuity controls within each frozen region.",
        },
        "existing_reviewed_regions": existing,
        "reserved_evaluation_regions": reserved,
        "regions": records,
        "leave_one_region_out": loro,
        "promotion_gate": {
            "requires_each_region": ["reviewed SAME_PROCESS > 0", "reviewed DIFFERENT_PROCESS > 0"],
            "requires_loro": "non-degenerate held-out-region boundary behavior before MV-GTVOL-000004 may be replayed",
            "protected_regression": _PROTECTED,
            "prohibited_actions": ["tune G1", "select checkpoints on MV-GTVOL-000004", "create MV-FRAG", "create MV-N", "create MV-CONN"],
        },
    }
    write_json_atomic(output_path, value)
    return value


def create_g2_review_inputs(*, cohort_path: Path, output_dir: Path) -> dict[str, Any]:
    """Derive immutable raw-only workspace input documents from a frozen cohort.

    This writes no machine labels and does not select review questions.  It is
    intentionally a bridge to the existing append-only external review API.
    """
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite G2 review inputs: {output_dir}")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if cohort.get("status") != "FROZEN_BEFORE_G1_DIAGNOSTIC_REVIEW":
        raise ValueError("G2 review inputs require a frozen pre-diagnostic cohort")
    output_dir.mkdir(parents=True)
    crops, requests = [], []
    for region in cohort["regions"]:
        raw = Path(region["raw_path"])
        if not raw.is_file() or sha256_file(raw) != region["raw_sha256"]:
            raise ValueError(f"{region['id']}: immutable raw source changed")
        crop_id = region["id"]
        origin = [region["bounds_xyz"][axis][0] for axis in ("x", "y", "z")]
        common = {
            "crop_id": crop_id,
            "parent_region_id": region["source_id"],
            "raw_sha256": region["raw_sha256"],
            "coordinate_frame": "MV-FRAME-DVID-WASP5-001",
            "source_origin_xyz": origin,
        }
        requests.append({**common, "role": region["role"]})
        crops.append({
            "id": crop_id, "parent_region_id": region["source_id"],
            "raw_crop_path": str(raw.resolve()), "raw_crop_sha256": region["raw_sha256"],
            "raw_shape_zyx": region["shape_zyx"], "raw_dtype": region["dtype"],
            "coordinate_frame": common["coordinate_frame"], "source_origin_xyz": origin,
            "bounds_xyz": region["bounds_xyz"], "split": region["role"],
            "status": "PENDING_FROZEN_G1_DIAGNOSTIC_QUEUE_AND_EXPERT_REVIEW",
        })
    request_path, manifest_path = output_dir / "request.json", output_dir / "crop-manifest.json"
    write_json_atomic(request_path, {"schema_version": 1, "id": "MV-G2-EXTERNAL-REVIEW-REQUEST-001", "cohort": {"path": str(cohort_path.resolve()), "sha256": sha256_file(cohort_path)}, "requested_review_sets": requests, "prohibited_sources": [_PROTECTED]})
    write_json_atomic(manifest_path, {"schema_version": 1, "id": "MV-G2-EXTERNAL-REVIEW-CROPS-001", "cohort": {"path": str(cohort_path.resolve()), "sha256": sha256_file(cohort_path)}, "crops": crops, "prohibited_sources": [_PROTECTED]})
    return {"request": str(request_path.resolve()), "manifest": str(manifest_path.resolve()), "regions": len(crops)}
