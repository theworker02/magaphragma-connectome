"""Immutable, split-safe DVID annotation crops and reviewed-label registration."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .io import sha256_file, write_json_atomic

IGNORE_LABEL = 2**32 - 1
REGRESSION_ONLY_PARENT = "MV-GTVOL-000004"


def _box(crop: dict) -> tuple[tuple[int, int], ...]:
    """Normalize one crop's half-open XYZ extent for split-safety comparisons."""
    return tuple((start, start + size) for start, size in zip(crop["origin_zyx"], crop["shape_zyx"], strict=True))


def _overlap(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> bool:
    """Return whether two half-open XYZ boxes share any voxel volume."""
    return all(max(left[0], right[0]) < min(left[1], right[1]) for left, right in zip(a, b, strict=True))


def validate_crop_plan(plan: dict, parents: dict[str, dict]) -> None:
    """Reject crops that violate parent, split, buffer, or regression boundaries."""
    identifiers = [crop["id"] for crop in plan["crops"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Annotation crop IDs must be unique")
    for crop in plan["crops"]:
        if crop["parent_region_id"] == REGRESSION_ONLY_PARENT:
            raise ValueError("Regression-only parent cannot supply annotation crops")
        if crop["split"] not in {"DVID_TARGET_TRAIN", "DVID_TARGET_VALIDATION"}:
            raise ValueError(f"{crop['id']}: invalid target split")
        parent = parents.get(crop["parent_region_id"])
        if parent is None:
            raise ValueError(f"{crop['id']}: missing parent region")
        if len(crop["origin_zyx"]) != 3 or len(crop["shape_zyx"]) != 3:
            raise ValueError(f"{crop['id']}: expected ZYX origin and shape")
        if any(value < 0 for value in crop["origin_zyx"]) or any(value <= 0 for value in crop["shape_zyx"]):
            raise ValueError(f"{crop['id']}: invalid crop bounds")
        parent_shape = tuple(reversed(parent["dimensions_voxels_xyz"]))
        if any(start + size > limit for start, size, limit in zip(crop["origin_zyx"], crop["shape_zyx"], parent_shape, strict=True)):
            raise ValueError(f"{crop['id']}: exceeds parent raw volume")
    for index, first in enumerate(plan["crops"]):
        for second in plan["crops"][index + 1:]:
            if first["parent_region_id"] == second["parent_region_id"] and first["split"] != second["split"] and _overlap(_box(first), _box(second)):
                raise ValueError(f"{first['id']} and {second['id']}: cross-split voxel overlap")


def materialize_annotation_crops(plan_path: Path, parent_manifest_path: Path, output_root: Path, crop_manifest_path: Path) -> dict:
    """Fetch planned raw DVID crops and write immutable local provenance records."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parents = {item["id"]: item for item in parent_manifest["regions"]}
    validate_crop_plan(plan, parents)
    if crop_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite crop manifest {crop_manifest_path}")
    records = []
    for crop in plan["crops"]:
        parent = parents[crop["parent_region_id"]]
        raw = np.load(parent["raw_path"], mmap_mode="r", allow_pickle=False)
        origin = tuple(crop["origin_zyx"])
        shape = tuple(crop["shape_zyx"])
        data = np.asarray(raw[tuple(slice(start, start + size) for start, size in zip(origin, shape, strict=True))]).copy()
        target = output_root / crop["id"] / "raw_zyx.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite immutable crop {target}")
        np.save(target, data, allow_pickle=False)
        parent_origin_xyz = [parent["bounds_xyz"][axis][0] for axis in ("x", "y", "z")]
        local_origin_xyz = [origin[2], origin[1], origin[0]]
        source_origin_xyz = [base + local for base, local in zip(parent_origin_xyz, local_origin_xyz, strict=True)]
        records.append({
            **crop,
            "annotation_status": "UNANNOTATED",
            "review_state": "NOT_STARTED",
            "parent_raw_sha256": parent["raw_sha256"],
            "raw_crop_path": str(target.resolve()), "raw_crop_sha256": sha256_file(target),
            "raw_dtype": str(data.dtype), "raw_shape_zyx": list(data.shape),
            "source_origin_xyz": source_origin_xyz,
            "voxel_size_nm_xyz": [8.0, 8.0, 8.0],
            "coordinate_frame": parent["coordinate_frame"], "dataset_id": parent["dataset_id"], "volume_id": parent["volume_id"],
            "specimen_status": "SPECIMEN_ID_UNKNOWN_DO_NOT_CROSS_SPECIMENS",
            "label_path": None, "label_sha256": None, "ignore_voxels": 0, "instance_count": 0,
            "machine_assist": None,
        })
    value = {"schema_version": 1, "id": plan["id"], "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
             "plan_sha256": sha256_file(plan_path), "status": "CROPS_MATERIALIZED_AWAITING_REVIEW", "crops": records}
    write_json_atomic(crop_manifest_path, value)
    return {"crops": len(records), "status": value["status"]}


def register_reviewed_crop_label(crop_manifest_path: Path, crop_id: str, label_path: Path, reviewer: str, receipt_path: Path) -> dict:
    """Copy a human-reviewed crop label immutably after fail-closed validation."""
    if not reviewer.strip():
        raise ValueError("Reviewer identity is required")
    manifest = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
    crop = next((item for item in manifest["crops"] if item["id"] == crop_id), None)
    if crop is None or crop["parent_region_id"] == REGRESSION_ONLY_PARENT:
        raise ValueError("Unknown or regression-only annotation crop")
    if crop["label_path"]:
        raise ValueError("Refusing to overwrite reviewed crop label")
    labels = np.load(label_path, allow_pickle=False)
    if labels.shape != tuple(crop["raw_shape_zyx"]) or labels.ndim != 3:
        raise ValueError("Reviewed label must exactly match raw crop ZYX shape")
    if labels.dtype.kind not in {"u", "i"} or (labels < 0).any():
        raise ValueError("Reviewed label must be non-negative integer IDs")
    instances = np.unique(labels[(labels != 0) & (labels != IGNORE_LABEL)])
    if not len(instances):
        raise ValueError("Reviewed DVID label requires at least one non-background, non-ignore instance")
    destination = receipt_path.parent / "labels" / f"{crop_id}-labels_zyx.npy"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable label {destination}")
    destination.write_bytes(label_path.read_bytes())
    crop.update({"annotation_status": "REVIEWED_DVID_LABEL", "review_state": "REVIEWED_DVID_LABEL", "reviewer": reviewer,
                 "label_path": str(destination.resolve()), "label_sha256": sha256_file(destination),
                 "instance_count": int(len(instances)), "ignore_voxels": int((labels == IGNORE_LABEL).sum())})
    receipt = {"id": f"MV-DVID-LABEL-RECEIPT-{crop_id[-6:]}", "crop_id": crop_id, "status": "REVIEWED_DVID_LABEL",
               "reviewer": reviewer, "raw_crop_sha256": crop["raw_crop_sha256"], "label_sha256": crop["label_sha256"],
               "shape_zyx": list(labels.shape), "instance_count": crop["instance_count"], "ignore_label": IGNORE_LABEL,
               "ignore_voxels": crop["ignore_voxels"], "split": crop["split"], "machine_proposal_promoted": False}
    write_json_atomic(crop_manifest_path, manifest)
    write_json_atomic(receipt_path, receipt)
    return receipt
