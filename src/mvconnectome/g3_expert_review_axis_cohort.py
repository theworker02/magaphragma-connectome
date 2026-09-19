"""Freeze an axis-balanced expert-review cohort from frozen G3 supervision.

This stage answers a single question that the upstream G3 freeze does not:
which Z/Y/X affinity axes are eligible to *evaluate* SAME-vs-DIFFERENT
representations without physical-axis leakage.

An affinity axis (channel 0=Z, 1=Y, 2=X) is only eligible when it carries at
least one reviewed SAME and at least one reviewed DIFFERENT pair in *both* the
TRAIN and the VALIDATION split -- all four (axis x split x class) cells must be
non-empty.  Any axis missing a cell is excluded and its deficient cells are
recorded; it is never silently retained.  At least one axis must be fully
eligible or the cohort fails closed.

Nothing here fabricates, duplicates, relabels, or otherwise manufactures
reviewed evidence.  The cohort is derived only from the already-frozen,
already-reviewed supervision receipts, it is immutable, it refuses to overwrite,
and it records full provenance (sha256) back to the source manifest and every
per-region receipt.  A failing dataset produces an auditable failure report
that states exactly which additional reviewed supervision must be collected.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, write_json_atomic

# Physical affinity axis per channel; kept explicit so the matrix labels can
# never drift from pair_supervision.OFFSETS_ZYX channel ordering.
_AXIS_NAME = {0: "Z", 1: "Y", 2: "X"}
_AXES = (0, 1, 2)
_CLASSES = ("SAME_PROCESS", "DIFFERENT_PROCESS")
_SPLITS = ("G3_TARGET_TRAIN", "G3_TARGET_VALIDATION")
_SPLIT_LABEL = {"G3_TARGET_TRAIN": "TRAIN", "G3_TARGET_VALIDATION": "VALIDATION"}

# The required non-empty cells for an axis to be evaluable.
_REQUIRED_CELLS = tuple((split, decision) for split in _SPLITS for decision in _CLASSES)

_SOURCE_STATUS = "FROZEN_READY_FOR_G3_LORO"


class AxisCohortFailure(RuntimeError):
    """Raised when no affinity axis satisfies the balance contract.

    The failure report has already been written when this is raised; the
    ``report`` attribute carries the same machine-readable structure.
    """

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        max(left["bounds_xyz"][axis][0], right["bounds_xyz"][axis][0])
        < min(left["bounds_xyz"][axis][1], right["bounds_xyz"][axis][1])
        for axis in ("x", "y", "z")
    )


def _empty_matrix() -> dict[int, dict[str, dict[str, int]]]:
    return {axis: {split: {decision: 0 for decision in _CLASSES} for split in _SPLITS} for axis in _AXES}


def _region_bounds(receipt: dict[str, Any], region: dict[str, Any]) -> dict[str, Any] | None:
    """Return spatial bounds for a region if the source manifest carried them.

    The freeze manifest does not always inline bounds_xyz; when present we
    re-check spatial disjointness so axis balance never comes at the cost of
    the existing TRAIN/VALIDATION voxel-leakage guarantee.  When absent we
    record that the invariant could not be re-verified here rather than
    silently claiming it held.
    """
    for source in (region, receipt):
        bounds = source.get("bounds_xyz")
        if isinstance(bounds, dict) and all(axis in bounds for axis in ("x", "y", "z")):
            return {"id": region["crop_id"], "bounds_xyz": bounds}
    return None


def _matrix_as_json(matrix: dict[int, dict[str, dict[str, int]]]) -> dict[str, Any]:
    return {
        str(axis): {
            "axis_name": _AXIS_NAME[axis],
            "cells": {
                _SPLIT_LABEL[split]: {decision: matrix[axis][split][decision] for decision in _CLASSES}
                for split in _SPLITS
            },
        }
        for axis in _AXES
    }


def _deficient_cells(matrix: dict[int, dict[str, dict[str, int]]], axis: int) -> list[dict[str, str]]:
    return [
        {"split": _SPLIT_LABEL[split], "class": decision}
        for split, decision in _REQUIRED_CELLS
        if matrix[axis][split][decision] == 0
    ]


def freeze_g3_expert_review_axis_cohort(
    *, supervision_manifest_path: Path, output_path: Path, cohort_id: str | None = None
) -> dict[str, Any]:
    """Freeze the axis-balanced expert-review cohort.

    Parameters
    ----------
    supervision_manifest_path:
        A frozen ``FROZEN_READY_FOR_G3_LORO`` manifest produced by
        ``freeze_g3_supervision``.
    output_path:
        Destination receipt path.  Refuses to overwrite an existing file so
        the artifact stays immutable; a replacement must be a new version.
    cohort_id:
        Optional explicit id; defaults to a deterministic id derived from the
        source manifest id.

    Returns the receipt on success.  Raises :class:`AxisCohortFailure` when no
    axis is eligible -- after writing the auditable failure report to
    ``output_path``.  Raises ``FileExistsError`` if ``output_path`` exists and
    ``ValueError`` for provenance/status violations.
    """
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable axis cohort: {output_path}")

    manifest = read_json(supervision_manifest_path)
    if manifest.get("status") != _SOURCE_STATUS:
        raise ValueError(
            f"Source is not a frozen G3 supervision manifest (status={manifest.get('status')!r}, "
            f"required {_SOURCE_STATUS!r})"
        )
    regions = manifest.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("Source manifest carries no regions")

    manifest_sha = sha256_file(supervision_manifest_path)
    resolved_id = cohort_id or f"MV-G3-AXIS-REVIEW-COHORT-{manifest.get('id', 'UNKNOWN')}"

    matrix = _empty_matrix()
    region_records: list[dict[str, Any]] = []
    bounds_seen: list[dict[str, Any]] = []
    bounds_missing: list[str] = []
    seen_crop_ids: set[str] = set()
    total_pairs = 0

    # Deterministic region order: by (split, crop_id) so the receipt is stable
    # regardless of manifest ordering.
    ordered_regions = sorted(
        regions, key=lambda item: (str(item.get("split")), str(item.get("crop_id")))
    )
    for region in ordered_regions:
        crop_id = region.get("crop_id")
        split = region.get("split")
        if crop_id is None or split not in _SPLITS:
            raise ValueError(f"Region has missing crop_id or unrecognized split: {region!r}")
        if crop_id in seen_crop_ids:
            raise ValueError(f"Duplicate crop_id in source manifest: {crop_id}")
        seen_crop_ids.add(crop_id)

        receipt_path = Path(region["receipt"])
        if not receipt_path.is_file():
            raise ValueError(f"{crop_id}: source region receipt does not exist: {receipt_path}")
        actual_sha = sha256_file(receipt_path)
        declared_sha = region.get("receipt_sha256")
        if declared_sha is not None and actual_sha != declared_sha:
            raise ValueError(f"{crop_id}: region receipt sha256 mismatch against source manifest")

        receipt = read_json(receipt_path)
        if receipt.get("crop_id") != crop_id or receipt.get("split") != split:
            raise ValueError(f"{crop_id}: region receipt crop/split disagrees with manifest")

        eligible_pairs = receipt.get("eligible_pairs")
        if not isinstance(eligible_pairs, list):
            raise ValueError(f"{crop_id}: region receipt lacks eligible_pairs list")

        per_region_matrix = {axis: {decision: 0 for decision in _CLASSES} for axis in _AXES}
        for pair in eligible_pairs:
            channel = int(pair["channel_zyx"])
            decision = pair["decision"]
            if channel not in _AXES or decision not in _CLASSES:
                raise ValueError(
                    f"{crop_id}: eligible pair has invalid channel/decision "
                    f"(channel={channel!r}, decision={decision!r})"
                )
            matrix[channel][split][decision] += 1
            per_region_matrix[channel][decision] += 1
            total_pairs += 1

        region_records.append(
            {
                "crop_id": crop_id,
                "split": split,
                "split_label": _SPLIT_LABEL[split],
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": actual_sha,
                "eligible_pair_count": len(eligible_pairs),
                "pairs_by_axis": {
                    _AXIS_NAME[axis]: dict(per_region_matrix[axis]) for axis in _AXES
                },
            }
        )

        located = _region_bounds(receipt, region)
        if located is None:
            bounds_missing.append(crop_id)
        else:
            bounds_seen.append(located)

    # Preserve the existing spatial-disjointness / voxel-leakage guarantee.
    # Axis balance is additive: it must never mask a TRAIN/VALIDATION overlap.
    spatial_overlaps: list[dict[str, str]] = []
    for i, left in enumerate(bounds_seen):
        for right in bounds_seen[i + 1:]:
            if _overlap(left, right):
                spatial_overlaps.append({"region_a": left["id"], "region_b": right["id"]})
    if spatial_overlaps:
        raise ValueError(
            "Spatial-disjointness invariant violated by source regions: "
            + ", ".join(f"{item['region_a']}/{item['region_b']}" for item in spatial_overlaps)
        )

    # Classify each axis against the 2x2 contract.
    included_axes: list[dict[str, Any]] = []
    excluded_axes: list[dict[str, Any]] = []
    for axis in _AXES:
        deficient = _deficient_cells(matrix, axis)
        axis_total = sum(
            matrix[axis][split][decision] for split, decision in _REQUIRED_CELLS
        )
        record = {
            "axis_channel_zyx": axis,
            "axis_name": _AXIS_NAME[axis],
            "total_reviewed_pairs": axis_total,
            "cells": {
                _SPLIT_LABEL[split]: {decision: matrix[axis][split][decision] for decision in _CLASSES}
                for split in _SPLITS
            },
        }
        if deficient:
            record["deficient_cells"] = deficient
            record["failure_reason"] = "AXIS_MISSING_REQUIRED_SPLIT_CLASS_CELL"
            record["required_additional_reviewed_supervision"] = [
                {
                    "axis_name": _AXIS_NAME[axis],
                    "split": cell["split"],
                    "class": cell["class"],
                    "needed_minimum_pairs": 1,
                }
                for cell in deficient
            ]
            excluded_axes.append(record)
        else:
            included_axes.append(record)

    matrix_json = _matrix_as_json(matrix)
    provenance = {
        "supervision_manifest": {
            "path": str(supervision_manifest_path.resolve()),
            "sha256": manifest_sha,
            "id": manifest.get("id"),
            "status": manifest.get("status"),
        },
        "source_region_receipts": [
            {"crop_id": record["crop_id"], "sha256": record["receipt_sha256"], "path": record["receipt_path"]}
            for record in region_records
        ],
    }
    spatial_guarantee = {
        "invariant": "TRAIN_VALIDATION_SPATIAL_DISJOINTNESS_PRESERVED",
        "method": "BOUNDS_XYZ_BOX_INTERSECTION",
        "regions_checked": [item["id"] for item in bounds_seen],
        "regions_without_bounds_not_recheckable_here": bounds_missing,
        "overlaps_detected": spatial_overlaps,
        "note": (
            "Axis balance is an additional constraint layered on top of the source "
            "G3 cohort validity conditions; it does not replace them. Regions listed "
            "as not re-checkable here were already spatially validated at G3 cohort "
            "freeze time; their bounds are simply not inlined in this manifest."
        ),
    }

    common = {
        "schema_version": 1,
        "id": resolved_id,
        "created_at": _now(),
        "cohort_kind": "G3_EXPERT_REVIEW_AXIS_BALANCED",
        "axis_eligibility_contract": {
            "definition": (
                "An affinity axis is eligible only if it carries at least one reviewed "
                "SAME_PROCESS and one reviewed DIFFERENT_PROCESS pair in BOTH the TRAIN "
                "and the VALIDATION split."
            ),
            "required_cells_per_axis": [
                {"split": _SPLIT_LABEL[split], "class": decision} for split, decision in _REQUIRED_CELLS
            ],
            "axis_channel_zyx_names": {str(axis): _AXIS_NAME[axis] for axis in _AXES},
        },
        "provenance": provenance,
        "spatial_disjointness_guarantee": spatial_guarantee,
        "region_summaries": region_records,
        "axis_split_class_matrix": matrix_json,
        "included_axes": included_axes,
        "excluded_axes": excluded_axes,
        "counts": {
            "regions": len(region_records),
            "total_reviewed_pairs": total_pairs,
            "eligible_axis_count": len(included_axes),
            "excluded_axis_count": len(excluded_axes),
        },
        "prohibited_promotions": ["MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
        "scientific_boundary": (
            "This cohort only declares which affinity axes may be evaluated for "
            "SAME/DIFFERENT representation separability without physical-axis leakage. "
            "It does not itself make biological claims and cannot enter training, labels, "
            "MV-FRAG, MV-N, MV-SYN, or MV-CONN."
        ),
    }

    if not included_axes:
        report = {
            **common,
            "status": "FAILED_NO_ELIGIBLE_AXIS",
            "failure_reason": "NO_AXIS_SATISFIES_TRAIN_AND_VALIDATION_SAME_AND_DIFFERENT",
            "remediation": (
                "Collect the additional reviewed supervision listed per excluded axis "
                "so that at least one Z/Y/X axis has both SAME and DIFFERENT reviewed "
                "pairs in both TRAIN and VALIDATION, then freeze a new cohort version."
            ),
        }
        write_json_atomic(output_path, report)
        raise AxisCohortFailure(
            "No affinity axis satisfies the TRAIN+VALIDATION SAME+DIFFERENT contract; "
            f"wrote failure report to {output_path}",
            report,
        )

    receipt = {**common, "status": "FROZEN_AXIS_BALANCED_EXPERT_REVIEW_COHORT"}
    write_json_atomic(output_path, receipt)
    return receipt
