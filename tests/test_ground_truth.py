"""Safety tests for DVID-native ground-truth registration and split isolation."""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from mvconnectome.ground_truth import register_instance_label, validate_plan
from mvconnectome.target_adaptation import adaptation_status


class GroundTruthPlanTests(unittest.TestCase):
    def test_cross_split_regions_require_a_real_spatial_buffer(self):
        plan = {"required_excluded_buffer_voxels": 10, "regions": [
            {"id": "MV-GTVOL-1", "split": "train", "bounds_xyz": {"x": [0, 10], "y": [0, 10], "z": [0, 10]}},
            {"id": "MV-GTVOL-2", "split": "test", "bounds_xyz": {"x": [15, 25], "y": [0, 10], "z": [0, 10]}},
        ]}
        with self.assertRaisesRegex(ValueError, "buffer"):
            validate_plan(plan)

    def test_spatially_separated_plan_is_valid(self):
        plan = {"required_excluded_buffer_voxels": 10, "regions": [
            {"id": "MV-GTVOL-1", "split": "train", "bounds_xyz": {"x": [0, 10], "y": [0, 10], "z": [0, 10]}},
            {"id": "MV-GTVOL-2", "split": "test", "bounds_xyz": {"x": [21, 31], "y": [0, 10], "z": [0, 10]}},
        ]}
        validate_plan(plan)

    def test_register_requires_matching_nonempty_instance_labels(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text('{"regions":[{"id":"MV-GTVOL-000001","dimensions_voxels_xyz":[2,2,2]}]}', encoding="utf-8")
            label = root / "label.npy"
            np.save(label, np.array([[[0, 1], [0, 1]], [[2, 2], [0, 0]]], dtype=np.uint32))
            result = register_instance_label(manifest, "MV-GTVOL-000001", label, "reviewer-1", "FIRST_PASS", root / "objects.json")
            self.assertEqual(result["instances"], 2)
            self.assertEqual(result["labeled_voxels"], 4)

    def test_regression_cube_cannot_be_registered_as_a_label(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text('{"regions":[{"id":"MV-GTVOL-000004","dimensions_voxels_xyz":[2,2,2]}]}', encoding="utf-8")
            label = root / "label.npy"
            np.save(label, np.ones((2, 2, 2), dtype=np.uint32))
            with self.assertRaisesRegex(ValueError, "regression-only"):
                register_instance_label(manifest, "MV-GTVOL-000004", label, "reviewer-1", "REVIEWED", root / "objects.json")

    def test_adaptation_requires_distinct_reviewed_train_and_validation_labels(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            plan.write_text('{"id":"plan","roles":{"train":["train"],"validation":["validation"],"regression_only":"regression"}}', encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text('{"regions":[{"id":"train","label_path":"a.npy","review_status":"REVIEWED"},{"id":"validation","label_path":null,"review_status":"UNANNOTATED"},{"id":"regression","label_path":null,"review_status":"UNANNOTATED"}]}', encoding="utf-8")
            result = adaptation_status(plan, manifest)
            self.assertEqual(result["status"], "BLOCKED_AWAITING_REVIEWED_DVID_NATIVE_LABELS")
