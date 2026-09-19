import json
from pathlib import Path

import pytest

from mvconnectome.g3_expert_review_axis_cohort import (
    AxisCohortFailure,
    freeze_g3_expert_review_axis_cohort,
)
from mvconnectome.io import sha256_file, write_json_atomic


def _pair(channel: int, decision: str, left: list[int]) -> dict:
    right = list(left)
    right[channel] += 1
    return {
        "interface_id": f"I-{channel}-{decision}-{left}",
        "channel_zyx": channel,
        "decision": decision,
        "pair_left_zyx": left,
        "pair_right_zyx": right,
    }


def _write_region_receipt(root: Path, crop_id: str, split: str, pairs: list[dict], *, bounds=None) -> Path:
    receipt = {
        "schema_version": 1,
        "id": f"MV-G3-REVIEWED-INTERFACE-AFFINITY-{crop_id}",
        "status": "REVIEWED_DVID_INTERFACE_AFFINITY_SUPERVISION",
        "crop_id": crop_id,
        "split": split,
        "eligible_pairs": pairs,
    }
    if bounds is not None:
        receipt["bounds_xyz"] = bounds
    path = root / f"{crop_id}-receipt.json"
    write_json_atomic(path, receipt)
    return path


def _write_manifest(root: Path, regions_spec: list[dict], *, manifest_id="MV-G3-SUP-001") -> Path:
    """regions_spec: list of {crop_id, split, pairs, [bounds]}."""
    regions = []
    for spec in regions_spec:
        receipt_path = _write_region_receipt(
            root, spec["crop_id"], spec["split"], spec["pairs"], bounds=spec.get("bounds")
        )
        entry = {
            "crop_id": spec["crop_id"],
            "split": spec["split"],
            "receipt": str(receipt_path.resolve()),
            "receipt_sha256": sha256_file(receipt_path),
            "eligible_pairs": {
                "SAME_PROCESS": sum(p["decision"] == "SAME_PROCESS" for p in spec["pairs"]),
                "DIFFERENT_PROCESS": sum(p["decision"] == "DIFFERENT_PROCESS" for p in spec["pairs"]),
            },
        }
        if spec.get("bounds") is not None:
            entry["bounds_xyz"] = spec["bounds"]
        regions.append(entry)
    manifest = {
        "schema_version": 1,
        "id": manifest_id,
        "status": "FROZEN_READY_FOR_G3_LORO",
        "regions": regions,
    }
    manifest_path = root / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest_path


def _fully_eligible_spec(bounds=True) -> list[dict]:
    """A TRAIN and a VALIDATION region that together make axis Z (0) eligible."""
    train_pairs = [
        _pair(0, "SAME_PROCESS", [1, 1, 1]),
        _pair(0, "DIFFERENT_PROCESS", [2, 2, 2]),
    ]
    val_pairs = [
        _pair(0, "SAME_PROCESS", [1, 1, 1]),
        _pair(0, "DIFFERENT_PROCESS", [2, 2, 2]),
    ]
    return [
        {
            "crop_id": "R-TRAIN",
            "split": "G3_TARGET_TRAIN",
            "pairs": train_pairs,
            "bounds": {"x": [0, 10], "y": [0, 10], "z": [0, 10]} if bounds else None,
        },
        {
            "crop_id": "R-VAL",
            "split": "G3_TARGET_VALIDATION",
            "pairs": val_pairs,
            "bounds": {"x": [20, 30], "y": [0, 10], "z": [0, 10]} if bounds else None,
        },
    ]


def test_passing_cohort_marks_axis_eligible(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, _fully_eligible_spec())
    receipt = freeze_g3_expert_review_axis_cohort(
        supervision_manifest_path=manifest, output_path=tmp_path / "cohort.json"
    )
    assert receipt["status"] == "FROZEN_AXIS_BALANCED_EXPERT_REVIEW_COHORT"
    assert receipt["counts"]["eligible_axis_count"] == 1
    eligible = receipt["included_axes"][0]
    assert eligible["axis_name"] == "Z"
    assert eligible["cells"]["TRAIN"]["SAME_PROCESS"] == 1
    assert eligible["cells"]["VALIDATION"]["DIFFERENT_PROCESS"] == 1
    # Y and X are excluded with all four cells deficient.
    excluded_names = {axis["axis_name"] for axis in receipt["excluded_axes"]}
    assert excluded_names == {"Y", "X"}
    y_axis = next(a for a in receipt["excluded_axes"] if a["axis_name"] == "Y")
    assert len(y_axis["deficient_cells"]) == 4
    assert y_axis["failure_reason"] == "AXIS_MISSING_REQUIRED_SPLIT_CLASS_CELL"


def test_deficient_cell_excludes_axis(tmp_path: Path) -> None:
    # Axis Z has SAME+DIFFERENT in TRAIN but only SAME in VALIDATION.
    spec = [
        {
            "crop_id": "R-TRAIN",
            "split": "G3_TARGET_TRAIN",
            "pairs": [_pair(0, "SAME_PROCESS", [1, 1, 1]), _pair(0, "DIFFERENT_PROCESS", [2, 2, 2])],
            "bounds": {"x": [0, 10], "y": [0, 10], "z": [0, 10]},
        },
        {
            "crop_id": "R-VAL",
            "split": "G3_TARGET_VALIDATION",
            "pairs": [_pair(0, "SAME_PROCESS", [1, 1, 1])],
            "bounds": {"x": [20, 30], "y": [0, 10], "z": [0, 10]},
        },
    ]
    manifest = _write_manifest(tmp_path, spec)
    with pytest.raises(AxisCohortFailure) as excinfo:
        freeze_g3_expert_review_axis_cohort(
            supervision_manifest_path=manifest, output_path=tmp_path / "cohort.json"
        )
    report = excinfo.value.report
    assert report["status"] == "FAILED_NO_ELIGIBLE_AXIS"
    z_axis = next(a for a in report["excluded_axes"] if a["axis_name"] == "Z")
    assert z_axis["deficient_cells"] == [{"split": "VALIDATION", "class": "DIFFERENT_PROCESS"}]
    needed = z_axis["required_additional_reviewed_supervision"]
    assert needed == [
        {"axis_name": "Z", "split": "VALIDATION", "class": "DIFFERENT_PROCESS", "needed_minimum_pairs": 1}
    ]
    # Failure report is still written to disk for audit.
    assert (tmp_path / "cohort.json").is_file()
    written = json.loads((tmp_path / "cohort.json").read_text())
    assert written["status"] == "FAILED_NO_ELIGIBLE_AXIS"


def test_no_eligible_axis_reports_full_matrix(tmp_path: Path) -> None:
    # Only SAME anywhere -> no axis can satisfy DIFFERENT cells.
    spec = [
        {
            "crop_id": "R-TRAIN",
            "split": "G3_TARGET_TRAIN",
            "pairs": [_pair(0, "SAME_PROCESS", [1, 1, 1])],
            "bounds": {"x": [0, 10], "y": [0, 10], "z": [0, 10]},
        },
        {
            "crop_id": "R-VAL",
            "split": "G3_TARGET_VALIDATION",
            "pairs": [_pair(0, "SAME_PROCESS", [1, 1, 1])],
            "bounds": {"x": [20, 30], "y": [0, 10], "z": [0, 10]},
        },
    ]
    manifest = _write_manifest(tmp_path, spec)
    with pytest.raises(AxisCohortFailure) as excinfo:
        freeze_g3_expert_review_axis_cohort(
            supervision_manifest_path=manifest, output_path=tmp_path / "cohort.json"
        )
    report = excinfo.value.report
    assert report["failure_reason"] == "NO_AXIS_SATISFIES_TRAIN_AND_VALIDATION_SAME_AND_DIFFERENT"
    # Complete matrix present for all three axes and both splits.
    matrix = report["axis_split_class_matrix"]
    assert set(matrix.keys()) == {"0", "1", "2"}
    assert matrix["0"]["cells"]["TRAIN"]["SAME_PROCESS"] == 1
    assert matrix["0"]["cells"]["VALIDATION"]["SAME_PROCESS"] == 1
    assert matrix["0"]["cells"]["TRAIN"]["DIFFERENT_PROCESS"] == 0
    assert report["counts"]["eligible_axis_count"] == 0


def test_deterministic_receipt_ignores_region_order(tmp_path: Path) -> None:
    spec = _fully_eligible_spec()
    out_a = tmp_path / "a" / "cohort.json"
    manifest_a = _write_manifest(tmp_path / "a", spec, manifest_id="MV-ID")
    receipt_a = freeze_g3_expert_review_axis_cohort(supervision_manifest_path=manifest_a, output_path=out_a)

    # Reverse region order; identical content otherwise.
    out_b = tmp_path / "b" / "cohort.json"
    manifest_b = _write_manifest(tmp_path / "b", list(reversed(spec)), manifest_id="MV-ID")
    receipt_b = freeze_g3_expert_review_axis_cohort(supervision_manifest_path=manifest_b, output_path=out_b)

    # Ignore volatile / path-bearing fields; the substantive cohort must match.
    for key in ("created_at", "provenance", "region_summaries"):
        receipt_a.pop(key)
        receipt_b.pop(key)
    assert receipt_a == receipt_b
    # Region summary ordering is deterministic (sorted by split, crop_id).
    order = [r["crop_id"] for r in json.loads(out_a.read_text())["region_summaries"]]
    assert order == ["R-TRAIN", "R-VAL"]


def test_provenance_records_manifest_and_receipt_hashes(tmp_path: Path) -> None:
    spec = _fully_eligible_spec()
    manifest = _write_manifest(tmp_path, spec)
    receipt = freeze_g3_expert_review_axis_cohort(
        supervision_manifest_path=manifest, output_path=tmp_path / "cohort.json"
    )
    prov = receipt["provenance"]
    assert prov["supervision_manifest"]["sha256"] == sha256_file(manifest)
    assert prov["supervision_manifest"]["status"] == "FROZEN_READY_FOR_G3_LORO"
    receipt_hashes = {r["crop_id"]: r["sha256"] for r in prov["source_region_receipts"]}
    assert set(receipt_hashes) == {"R-TRAIN", "R-VAL"}
    for crop_id, digest in receipt_hashes.items():
        assert digest == sha256_file(tmp_path / f"{crop_id}-receipt.json")


def test_receipt_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    spec = _fully_eligible_spec()
    manifest_path = _write_manifest(tmp_path, spec)
    manifest = json.loads(manifest_path.read_text())
    manifest["regions"][0]["receipt_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="receipt sha256 mismatch"):
        freeze_g3_expert_review_axis_cohort(
            supervision_manifest_path=manifest_path, output_path=tmp_path / "cohort.json"
        )


def test_spatial_overlap_is_rejected(tmp_path: Path) -> None:
    # TRAIN and VALIDATION bounds overlap -> voxel leakage guard must fire.
    spec = _fully_eligible_spec()
    spec[1]["bounds"] = {"x": [5, 15], "y": [0, 10], "z": [0, 10]}  # overlaps R-TRAIN x[0,10]
    manifest = _write_manifest(tmp_path, spec)
    with pytest.raises(ValueError, match="Spatial-disjointness"):
        freeze_g3_expert_review_axis_cohort(
            supervision_manifest_path=manifest, output_path=tmp_path / "cohort.json"
        )


def test_missing_bounds_recorded_not_silently_passed(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, _fully_eligible_spec(bounds=False))
    receipt = freeze_g3_expert_review_axis_cohort(
        supervision_manifest_path=manifest, output_path=tmp_path / "cohort.json"
    )
    guarantee = receipt["spatial_disjointness_guarantee"]
    assert set(guarantee["regions_without_bounds_not_recheckable_here"]) == {"R-TRAIN", "R-VAL"}
    assert guarantee["overlaps_detected"] == []


def test_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, _fully_eligible_spec())
    output = tmp_path / "cohort.json"
    freeze_g3_expert_review_axis_cohort(supervision_manifest_path=manifest, output_path=output)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        freeze_g3_expert_review_axis_cohort(supervision_manifest_path=manifest, output_path=output)


def test_rejects_non_frozen_source_status(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, _fully_eligible_spec())
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "SOMETHING_ELSE"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not a frozen G3 supervision manifest"):
        freeze_g3_expert_review_axis_cohort(
            supervision_manifest_path=manifest_path, output_path=tmp_path / "cohort.json"
        )
