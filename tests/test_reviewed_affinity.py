"""Regression coverage for reviewed decisions becoming masked affinity targets."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mvconnectome.proofreading import append_review_event, create_workspace
from mvconnectome.reviewed_affinity import materialize_reviewed_affinity


class ReviewedAffinityTests(unittest.TestCase):
    def test_reviewed_same_and_different_become_masked_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw = root / "raw.npy"; labels = root / "labels.npy"; manifest = root / "manifest.json"; workspace = root / "workspace.json"
            np.save(raw, np.zeros((3, 3, 5), dtype=np.uint8)); np.save(labels, np.array([[[1, 1, 2, 2, 3]] * 3] * 3, dtype=np.uint32))
            manifest.write_text(json.dumps({"crops": [{"id": "MV-DVID-ANN-000001", "parent_region_id": "MV-GTVOL-000001", "raw_shape_zyx": [3, 3, 5], "raw_crop_path": str(raw), "raw_crop_sha256": "raw", "coordinate_frame": "frame", "source_origin_xyz": [0, 0, 0]}]}), encoding="utf-8")
            create_workspace(crop_manifest_path=manifest, crop_id="MV-DVID-ANN-000001", supervoxels_path=labels, output=workspace, source_method="test", source_run_id="run")
            append_review_event(workspace, event={"kind": "SAME_PROCESS", "reviewer": "human", "candidate": {"id": "a", "supervoxel_a": 1, "supervoxel_b": 2, "local_point_zyx": [1, 1, 1]}})
            append_review_event(workspace, event={"kind": "DIFFERENT_PROCESS", "reviewer": "human", "candidate": {"id": "b", "supervoxel_a": 2, "supervoxel_b": 3, "local_point_zyx": [1, 1, 3]}})
            result = materialize_reviewed_affinity(workspace, root / "output")
            self.assertEqual(result["counts"]["SAME_PROCESS"], 1); self.assertEqual(result["counts"]["DIFFERENT_PROCESS"], 1)
            self.assertEqual(result["counts"]["mask_pairs"], 2)

    def test_queue_ordinal_does_not_collapse_distinct_interfaces(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw = root / "raw.npy"; labels = root / "labels.npy"; manifest = root / "manifest.json"; workspace = root / "workspace.json"
            np.save(raw, np.zeros((3, 3, 6), dtype=np.uint8)); np.save(labels, np.array([[[1, 1, 2, 2, 3, 3]] * 3] * 3, dtype=np.uint32))
            manifest.write_text(json.dumps({"crops": [{"id": "MV-DVID-ANN-000001", "parent_region_id": "MV-GTVOL-000001", "raw_shape_zyx": [3, 3, 6], "raw_crop_path": str(raw), "raw_crop_sha256": "raw", "coordinate_frame": "frame", "source_origin_xyz": [0, 0, 0]}]}), encoding="utf-8")
            create_workspace(crop_manifest_path=manifest, crop_id="MV-DVID-ANN-000001", supervoxels_path=labels, output=workspace, source_method="test", source_run_id="run")
            append_review_event(workspace, event={"kind": "SAME_PROCESS", "reviewer": "human", "candidate": {"id": "queue-a-0001", "supervoxel_a": 1, "supervoxel_b": 2, "local_point_zyx": [1, 1, 1]}})
            append_review_event(workspace, event={"kind": "DIFFERENT_PROCESS", "reviewer": "human", "candidate": {"id": "queue-b-0001", "supervoxel_a": 2, "supervoxel_b": 3, "local_point_zyx": [1, 1, 3]}})
            result = materialize_reviewed_affinity(workspace, root / "output")
            self.assertEqual(result["counts"]["mask_pairs"], 2)
