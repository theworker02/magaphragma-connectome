"""Regression coverage for spatial isolation before G2 review and adaptation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mvconnectome.io import sha256_file
from mvconnectome.spatial_expansion import create_g2_review_inputs, freeze_spatial_expansion_cohort


def _record(root: Path, identifier: str, x: int) -> dict:
    raw = root / f"{identifier}.npy"
    np.save(raw, np.zeros((3, 4, 5), dtype=np.uint8))
    return {"id": identifier, "bounds_xyz": {"x": [x, x + 5], "y": [0, 4], "z": [0, 3]},
            "raw_path": str(raw), "raw_sha256": sha256_file(raw), "shape_zyx": [3, 4, 5], "dtype": "uint8", "source_url": "https://example.invalid/raw"}


def test_freezes_non_overlapping_loro_cohort() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        survey = root / "survey.json"; plan = root / "plan.json"; output = root / "cohort.json"
        survey.write_text(json.dumps({"source": {"voxel_size_nm_xyz": [8, 8, 8]}, "records": [_record(root, "A", 0), _record(root, "B", 20), _record(root, "C", 40)]}))
        plan.write_text(json.dumps({"selected_regions": [{"source_id": "A", "role": "G2_TARGET_TRAIN"}, {"source_id": "B", "role": "G2_TARGET_TRAIN"}, {"source_id": "C", "role": "G2_TARGET_VALIDATION"}], "existing_reviewed_regions": []}))
        result = freeze_spatial_expansion_cohort(survey_manifest_path=survey, plan_path=plan, output_path=output)
        assert len(result["leave_one_region_out"]) == 2
        assert result["status"] == "FROZEN_BEFORE_G1_DIAGNOSTIC_REVIEW"
        inputs = create_g2_review_inputs(cohort_path=output, output_dir=root / "review-inputs")
        assert inputs["regions"] == 3
        request = json.loads(Path(inputs["request"]).read_text())
        assert request["requested_review_sets"][0]["crop_id"] == "MV-G2-TRAIN-001"


def test_rejects_overlap_with_existing_reviewed_region() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        survey = root / "survey.json"; plan = root / "plan.json"
        survey.write_text(json.dumps({"source": {"voxel_size_nm_xyz": [8, 8, 8]}, "records": [_record(root, "A", 0), _record(root, "B", 20)]}))
        plan.write_text(json.dumps({"selected_regions": [{"source_id": "A", "role": "G2_TARGET_TRAIN"}, {"source_id": "B", "role": "G2_TARGET_VALIDATION"}], "existing_reviewed_regions": [{"id": "old", "bounds_xyz": {"x": [2, 7], "y": [0, 4], "z": [0, 3]}}]}))
        try:
            freeze_spatial_expansion_cohort(survey_manifest_path=survey, plan_path=plan, output_path=root / "out.json")
        except ValueError as error:
            assert "overlaps" in str(error)
        else:
            raise AssertionError("expected overlap rejection")
