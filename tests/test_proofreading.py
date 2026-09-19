"""Regression coverage for append-only machine-only proofreading workflows."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mvconnectome.proofreading import append_review_event, create_workspace, rank_boundary_candidates


class ProofreadingTests(unittest.TestCase):
    def test_workspace_and_append_only_events_do_not_promote_machine_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw.npy"; labels = root / "sv.npy"; manifest = root / "crops.json"; workspace = root / "workspace.json"
            np.save(raw, np.zeros((3, 4, 5), dtype=np.uint8)); np.save(labels, np.array([[[1, 1, 2, 2, 0]] * 4] * 3, dtype=np.uint32))
            manifest.write_text(json.dumps({"crops": [{"id": "MV-DVID-ANN-000001", "parent_region_id": "MV-GTVOL-000001", "raw_shape_zyx": [3, 4, 5], "raw_crop_path": str(raw), "raw_crop_sha256": "raw", "coordinate_frame": "frame", "source_origin_xyz": [0, 0, 0]}]}), encoding="utf-8")
            value = create_workspace(crop_manifest_path=manifest, crop_id="MV-DVID-ANN-000001", supervoxels_path=labels, output=workspace, source_method="true-3d watershed", source_run_id="run")
            self.assertEqual(value["status"], "MACHINE_SUPERVOXEL_GRAPH_REVIEW_REQUIRED")
            event = append_review_event(workspace, event={"kind": "MERGE", "reviewer": "reviewer", "supervoxel_ids": [1, 2]})
            self.assertEqual(event["effect"], "REVIEW_DECISION_RECORDED_NOT_BIOLOGICAL_PROMOTION")
            self.assertTrue(Path(value["edit_log"]["path"]).exists())
            decision = append_review_event(workspace, event={"kind": "DIFFERENT_PROCESS", "reviewer": "reviewer", "candidate": {"supervoxel_a": 1, "supervoxel_b": 2}})
            self.assertEqual(decision["kind"], "DIFFERENT_PROCESS")

    def test_ranking_creates_questions_not_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw = root / "raw.npy"; labels = root / "sv.npy"; affinity = root / "aff.npy"; manifest = root / "crops.json"; workspace = root / "workspace.json"; queue = root / "queue.json"
            np.save(raw, np.zeros((3, 4, 5), dtype=np.uint8)); np.save(labels, np.array([[[1, 1, 2, 2, 0]] * 4] * 3, dtype=np.uint32)); np.save(affinity, np.full((3, 3, 4, 5), 0.5, dtype=np.float32))
            manifest.write_text(json.dumps({"crops": [{"id": "MV-DVID-ANN-000001", "parent_region_id": "MV-GTVOL-000001", "raw_shape_zyx": [3, 4, 5], "raw_crop_path": str(raw), "raw_crop_sha256": "raw", "coordinate_frame": "frame", "source_origin_xyz": [0, 0, 0]}]}), encoding="utf-8")
            create_workspace(crop_manifest_path=manifest, crop_id="MV-DVID-ANN-000001", supervoxels_path=labels, output=workspace, source_method="test", source_run_id="run")
            result = rank_boundary_candidates(workspace_path=workspace, affinity_path=affinity, output=queue)
            self.assertGreater(result["candidates"], 0)
            self.assertEqual(json.loads(queue.read_text())["candidates"][0]["status"], "PROOFREADING_CANDIDATE_MACHINE_RANKED_NOT_A_DECISION")

    def test_low_affinity_queue_is_explicitly_counterevidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw = root / "raw.npy"; labels = root / "sv.npy"; affinity = root / "aff.npy"; manifest = root / "crops.json"; workspace = root / "workspace.json"; queue = root / "queue.json"
            np.save(raw, np.zeros((3, 4, 5), dtype=np.uint8)); np.save(labels, np.array([[[1, 1, 2, 2, 0]] * 4] * 3, dtype=np.uint32)); np.save(affinity, np.full((3, 3, 4, 5), 0.1, dtype=np.float32))
            manifest.write_text(json.dumps({"crops": [{"id": "MV-DVID-ANN-000001", "parent_region_id": "MV-GTVOL-000001", "raw_shape_zyx": [3, 4, 5], "raw_crop_path": str(raw), "raw_crop_sha256": "raw", "coordinate_frame": "frame", "source_origin_xyz": [0, 0, 0]}]}), encoding="utf-8")
            create_workspace(crop_manifest_path=manifest, crop_id="MV-DVID-ANN-000001", supervoxels_path=labels, output=workspace, source_method="test", source_run_id="run")
            rank_boundary_candidates(workspace_path=workspace, affinity_path=affinity, output=queue, strategy="low_affinity")
            self.assertEqual(json.loads(queue.read_text())["candidates"][0]["selection_strategy"], "low_affinity")

    def test_regression_only_crop_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw = root / "raw.npy"; labels = root / "sv.npy"; manifest = root / "crops.json"
            np.save(raw, np.zeros((2, 2, 2), dtype=np.uint8)); np.save(labels, np.ones((2, 2, 2), dtype=np.uint32))
            manifest.write_text(json.dumps({"crops": [{"id": "MV-DVID-ANN-000004", "parent_region_id": "MV-GTVOL-000004", "raw_shape_zyx": [2, 2, 2], "raw_crop_path": str(raw), "raw_crop_sha256": "raw", "coordinate_frame": "frame", "source_origin_xyz": [0, 0, 0]}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                create_workspace(crop_manifest_path=manifest, crop_id="MV-DVID-ANN-000004", supervoxels_path=labels, output=root / "workspace.json", source_method="test", source_run_id="test")
