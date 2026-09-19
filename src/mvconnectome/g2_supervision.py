"""Freeze and validate the reviewed spatial G2 affinity supervision dataset.

This is intentionally stricter than per-crop materialization: it validates the
seven-region split, excludes crop-edge pairs from *use* (without mutating the
append-only source evidence), and creates a single immutable training manifest.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import sha256_file, write_json_atomic


_TRAIN = "G2_TARGET_TRAIN"
_VALIDATION = "G2_TARGET_VALIDATION"
_PLACEHOLDERS = {"expert-identifier", "reviewer", "unknown", "tbd", "to-be-assigned"}


def _now() -> str:
    """Return a UTC timestamp for the frozen supervision receipt."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _boxes_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Detect half-open spatial overlap between candidate reviewed regions."""
    return all(
        max(left["bounds_xyz"][axis][0], right["bounds_xyz"][axis][0])
        < min(left["bounds_xyz"][axis][1], right["bounds_xyz"][axis][1])
        for axis in ("x", "y", "z")
    )


def _is_interior(event: dict[str, Any], shape_zyx: tuple[int, int, int]) -> bool:
    """Exclude affinity pairs touching a crop edge from supervised use."""
    left = tuple(int(v) for v in event["pair_left_zyx"])
    right = list(left)
    right[int(event["channel_zyx"])] += 1
    return all(0 < value < limit - 1 for value, limit in zip(left, shape_zyx, strict=True)) and all(
        0 < value < limit - 1 for value, limit in zip(right, shape_zyx, strict=True)
    )


def freeze_g2_supervision(*, cohort_path: Path, receipt_root: Path, output_path: Path) -> dict[str, Any]:
    """Fail closed unless all frozen G2 regions retain reviewed interior classes."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen G2 dataset: {output_path}")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    regions = cohort.get("regions", [])
    if len(regions) != 7:
        raise ValueError("Frozen G2 cohort must contain exactly seven spatial regions")
    ids = {str(region["id"]) for region in regions}
    if ids != {"MV-G2-TRAIN-A", "MV-G2-TRAIN-B", "MV-G2-TRAIN-C", "MV-G2-TRAIN-D", "MV-G2-VALIDATION-V1", "MV-G2-VALIDATION-V2", "MV-G2-VALIDATION-V3"}:
        raise ValueError("Frozen cohort IDs do not match the required G2 design")
    for index, left in enumerate(regions):
        for right in regions[index + 1:]:
            if _boxes_overlap(left, right):
                raise ValueError(f"G2 split contamination: {left['id']} overlaps {right['id']}")
    # The cohort names the protected volume in its policy text.  Only a source
    # record could contaminate supervision; policy references are required.
    supervised_records = [*regions, *cohort.get("existing_reviewed_regions", [])]
    if any("000004" in json.dumps(record, sort_keys=True) for record in supervised_records):
        raise ValueError("Protected MV-GTVOL-000004 is prohibited from G2 supervision")

    admitted: list[dict[str, Any]] = []
    for region in regions:
        crop_id = str(region["id"])
        receipt_path = receipt_root / crop_id / "receipt.json"
        if not receipt_path.is_file():
            raise ValueError(f"{crop_id}: missing immutable reviewed-affinity receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "REVIEWED_DVID_AFFINITY_SUPERVISION" or receipt.get("crop_id") != crop_id:
            raise ValueError(f"{crop_id}: invalid reviewed-affinity receipt")
        if receipt.get("split") != region["role"] or receipt.get("raw_sha256") != region["raw_sha256"]:
            raise ValueError(f"{crop_id}: split or immutable raw hash mismatch")
        shape = tuple(int(v) for v in receipt["targets"]["shape_czyx"][1:])
        if tuple(int(v) for v in region["shape_zyx"]) != shape:
            raise ValueError(f"{crop_id}: supervision tensor geometry differs from frozen crop")
        for item in ("targets", "mask"):
            artifact = Path(receipt[item]["path"])
            if not artifact.is_file() or sha256_file(artifact) != receipt[item]["sha256"]:
                raise ValueError(f"{crop_id}: {item} artifact hash mismatch")
        log = Path(receipt["event_log"]["path"])
        if not log.is_file() or sha256_file(log) != receipt["event_log"]["sha256"]:
            raise ValueError(f"{crop_id}: frozen review log mutated")
        seen: set[tuple[int, tuple[int, int, int]]] = set()
        usable: list[dict[str, Any]] = []
        excluded_edge = 0
        for event in receipt.get("effective_decisions", []):
            if event.get("reviewer", "").strip().casefold() in _PLACEHOLDERS:
                raise ValueError(f"{crop_id}: placeholder reviewer entered effective supervision")
            channel, left = int(event["channel_zyx"]), tuple(int(v) for v in event["pair_left_zyx"])
            if channel not in (0, 1, 2) or len(left) != 3:
                raise ValueError(f"{crop_id}: invalid affinity geometry")
            right = list(left); right[channel] += 1
            if any(v < 0 or v >= limit for v, limit in zip(left, shape, strict=True)) or right[channel] >= shape[channel]:
                raise ValueError(f"{crop_id}: out-of-bounds reviewed pair")
            key = (channel, left)
            if key in seen:
                raise ValueError(f"{crop_id}: duplicate or contradictory canonical pair")
            seen.add(key)
            if not _is_interior(event, shape):
                excluded_edge += 1
                continue
            usable.append(event)
        counts = {decision: sum(event["decision"] == decision for event in usable) for decision in ("SAME_PROCESS", "DIFFERENT_PROCESS")}
        if not counts["SAME_PROCESS"] or not counts["DIFFERENT_PROCESS"]:
            raise ValueError(f"{crop_id}: edge exclusion leaves insufficient reviewed SAME/DIFFERENT evidence")
        admitted.append({
            "crop_id": crop_id, "split": region["role"],
            "raw": {"path": region["raw_path"], "sha256": region["raw_sha256"], "shape_zyx": region["shape_zyx"]},
            "raw_sha256": region["raw_sha256"],
            "receipt": str(receipt_path.resolve()), "receipt_sha256": sha256_file(receipt_path),
            "review_log": str(log.resolve()), "review_log_sha256": sha256_file(log),
            "targets": receipt["targets"], "mask": receipt["mask"],
            "usable_interior_pairs": usable, "usable_counts": counts,
            "excluded_edge_pairs": excluded_edge,
            "note": "Crop-edge decisions remain in the immutable append-only log but are IGNORE for G2 fitting/evaluation.",
        })
    result = {
        "schema_version": 1, "id": output_path.stem, "created_at": _now(),
        "status": "FROZEN_READY_FOR_G2_LORO", "cohort": {"path": str(cohort_path.resolve()), "sha256": sha256_file(cohort_path)},
        "regions": admitted,
        "loro_folds": cohort["leave_one_region_out"],
        "protected_regression": {"volume": "MV-GTVOL-000004", "included": False},
        "training_boundary": "Only G2_TARGET_TRAIN usable_interior_pairs may enter G2 optimization. Validation regions are evaluation-only.",
        "biological_promotions": {"MV-FRAG": 0, "MV-N": 0, "MV-SYN": 0, "MV-CONN": 0},
    }
    write_json_atomic(output_path, result)
    return result
