"""Regression coverage for raw-EM membrane review without biological promotion."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mvconnectome.io import sha256_file
from mvconnectome.membrane_pilot import append_membrane_review_event, create_membrane_review_workspace, materialize_reviewed_membrane_affinity


class MembranePilotTests(unittest.TestCase):
    def test_raw_pair_workspace_is_append_only_and_never_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw = root / "raw.npy"; receipt = root / "raw-receipt.json"; questions = root / "questions.json"; workspace = root / "workspace.json"
            np.save(raw, np.zeros((4, 5, 6), dtype=np.uint8))
            receipt.write_text(json.dumps({"id": "MV-DVID-ANN-000005-MEMBRANE-PILOT-001", "parent_region_id": "MV-GTVOL-000005", "coordinate_frame": "frame", "raw_path": str(raw), "raw_sha256": sha256_file(raw), "crop_shape_zyx": [4, 5, 6]}), encoding="utf-8")
            questions.write_text(json.dumps({"id": "MV-DVID-ANN-000005-MEMBRANE-PILOT-001", "raw_sha256": sha256_file(raw), "status": "MEMBRANE_CANDIDATE_REVIEW_REQUIRED", "algorithm": {"name": "test"}, "questions": [{"id": "Q1", "center_zyx": [1, 2, 3], "a_zyx": [1, 2, 3], "b_zyx": [1, 2, 4], "normal_zyx": [0, 0, 1], "affinity_channel_zyx": 2, "raw_feature_score": 1.0, "orientation_confidence": 1.0, "neighborhood_quality": "test"}]}), encoding="utf-8")
            value = create_membrane_review_workspace(raw_receipt=receipt, questions=questions, output=workspace)
            self.assertEqual(value["status"], "RAW_EM_MEMBRANE_REVIEW_REQUIRED")
            event = append_membrane_review_event(workspace, reviewer="reviewer", question_id="Q1", decision="DIFFERENT_PROCESS")
            self.assertEqual(event["effect"], "REVIEWED_LOCAL_AFFINITY_EVIDENCE_NOT_A_BIOLOGICAL_PROMOTION")
            self.assertTrue(Path(value["decision_log"]["path"]).exists())
            with self.assertRaises(ValueError):
                append_membrane_review_event(workspace, reviewer="reviewer", question_id="Q1", decision="SAME_PROCESS")

    def test_regression_parent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw = root / "raw.npy"; receipt = root / "raw-receipt.json"; questions = root / "questions.json"
            np.save(raw, np.zeros((2, 2, 2), dtype=np.uint8))
            receipt.write_text(json.dumps({"id": "pilot", "parent_region_id": "MV-GTVOL-000004", "coordinate_frame": "frame", "raw_path": str(raw), "raw_sha256": sha256_file(raw), "crop_shape_zyx": [2, 2, 2]}), encoding="utf-8")
            questions.write_text(json.dumps({"id": "pilot", "raw_sha256": sha256_file(raw), "status": "MEMBRANE_CANDIDATE_REVIEW_REQUIRED", "questions": [{"id": "Q1"}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                create_membrane_review_workspace(raw_receipt=receipt, questions=questions, output=root / "workspace.json")

    def test_continuity_candidate_queue_can_use_the_same_append_only_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw = root / "raw.npy"; receipt = root / "raw-receipt.json"; questions = root / "questions.json"; workspace = root / "workspace.json"
            np.save(raw, np.zeros((4, 5, 6), dtype=np.uint8))
            receipt.write_text(json.dumps({"id": "continuity-pilot", "parent_region_id": "MV-GTVOL-000001", "coordinate_frame": "frame", "raw_path": str(raw), "raw_sha256": sha256_file(raw), "crop_shape_zyx": [4, 5, 6]}), encoding="utf-8")
            questions.write_text(json.dumps({"id": "continuity-pilot", "raw_sha256": sha256_file(raw), "status": "CONTINUITY_CANDIDATE_REVIEW_REQUIRED", "algorithm": {"name": "test"}, "questions": [{"id": "Q1", "center_zyx": [1, 2, 3], "a_zyx": [1, 2, 3], "b_zyx": [1, 2, 4], "normal_zyx": [0, 0, 1], "affinity_channel_zyx": 2, "raw_feature_score": 1.0, "orientation_confidence": None, "neighborhood_quality": "test"}]}), encoding="utf-8")
            value = create_membrane_review_workspace(raw_receipt=receipt, questions=questions, output=workspace)
            self.assertEqual(value["status"], "RAW_EM_MEMBRANE_REVIEW_REQUIRED")

    def test_reviewed_pairs_materialize_to_masked_affinity_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw = root / "raw.npy"; receipt = root / "raw-receipt.json"; questions = root / "questions.json"; workspace = root / "workspace.json"
            np.save(raw, np.zeros((4, 5, 6), dtype=np.uint8))
            receipt.write_text(json.dumps({"id": "pilot", "parent_region_id": "MV-GTVOL-000001", "coordinate_frame": "frame", "raw_path": str(raw), "raw_sha256": sha256_file(raw), "crop_shape_zyx": [4, 5, 6]}), encoding="utf-8")
            questions.write_text(json.dumps({"id": "pilot", "raw_sha256": sha256_file(raw), "status": "MEMBRANE_CANDIDATE_REVIEW_REQUIRED", "questions": [
                {"id": "Q1", "center_zyx": [1, 2, 3], "a_zyx": [1, 2, 3], "b_zyx": [1, 2, 4], "normal_zyx": [0, 0, 1], "affinity_channel_zyx": 2, "raw_feature_score": 1.0, "orientation_confidence": 1.0, "neighborhood_quality": "test"},
                {"id": "Q2", "center_zyx": [1, 1, 1], "a_zyx": [1, 1, 1], "b_zyx": [2, 1, 1], "normal_zyx": [1, 0, 0], "affinity_channel_zyx": 0, "raw_feature_score": 1.0, "orientation_confidence": 1.0, "neighborhood_quality": "test"}
            ]}), encoding="utf-8")
            create_membrane_review_workspace(raw_receipt=receipt, questions=questions, output=workspace)
            append_membrane_review_event(workspace, reviewer="em-reviewer", question_id="Q1", decision="DIFFERENT_PROCESS")
            append_membrane_review_event(workspace, reviewer="em-reviewer", question_id="Q2", decision="SAME_PROCESS")
            result = materialize_reviewed_membrane_affinity(workspace_paths=[workspace], role="TEST", output=root / "out")
            self.assertEqual(result["counts"]["SAME_PROCESS"], 1)
            self.assertEqual(result["counts"]["DIFFERENT_PROCESS"], 1)
            self.assertEqual(result["counts"]["effective_pairs"], 2)
