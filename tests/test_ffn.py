"""Contract tests for the quarantined, validation-gated FFN adapter."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mvconnectome.ffn import build_seed_and_holdout_manifests, import_ffn_labels, plan_subvolumes, validate_ffn_run


class FfnTests(unittest.TestCase):
    def _ingestion(self, root: Path) -> Path:
        raw = root / "raw.npy"; np.save(raw, np.zeros((4, 8, 8), dtype=np.uint8))
        path = root / "ingestion.json"
        path.write_text(json.dumps({"raw_array": str(raw), "raw_sha256": "a" * 64, "volume_id": "MV-VOL-TEST", "region_id": "MV-REG-TEST", "source_bounds_xyz": {"x": [0, 8], "y": [0, 8], "z": [0, 4]}}))
        return path

    def test_plan_has_explicit_core_and_halo_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "plan.json"
            plan_subvolumes(self._ingestion(root), output, (4, 4, 2), (1, 1, 1))
            plan = json.loads(output.read_text())
            self.assertEqual(len(plan["work_units"]), 8)
            self.assertEqual(plan["work_units"][0]["core_global_voxel_bounds_xyz"], [[0, 4], [0, 4], [0, 2]])

    def test_validation_blocks_merge_and_accepts_clean_sparse_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); labels = root / "labels.npy"; np.save(labels, np.array([[[1, 1, 0], [0, 2, 2]]], dtype=np.uint32))
            heldout = root / "heldout.json"
            heldout.write_text(json.dumps({"registration": {"id": "MV-MAP-TEST"}, "records": [
                {"mv_neuron_id": "MV-N-000001", "reference_path_voxels_zyx": [[0, 0, 0], [0, 0, 1]]},
                {"mv_neuron_id": "MV-N-000002", "reference_path_voxels_zyx": [[0, 1, 1], [0, 1, 2]]}
            ]}))
            output = root / "validation.json"; validate_ffn_run(labels, heldout, output)
            self.assertEqual(json.loads(output.read_text())["candidate_extension_gate"], "PASSED")
            np.save(labels, np.array([[[1, 1, 0], [0, 1, 1]]], dtype=np.uint32))
            validate_ffn_run(labels, heldout, output)
            self.assertEqual(json.loads(output.read_text())["candidate_extension_gate"], "BLOCKED")

    def test_seed_export_requires_verified_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ingestion = self._ingestion(root)
            nodes = root / "nodes.parquet"; pq.write_table(pa.Table.from_pylist([{ "mv_neuron_id": "MV-N-000001", "source_treenode_id": 1, "parent_treenode_id": None, "x_nm": 1.0, "y_nm": 1.0, "z_nm": 1.0 }]), nodes)
            physical = root / "physical.json"; physical.write_text(json.dumps({"coordinate_frame": {"id": "MV-FRAME-CATMAID-001"}}))
            registration = root / "registration.json"; registration.write_text(json.dumps({"id": "MV-MAP-TEST", "source_frame_id": "MV-FRAME-CATMAID-001", "target_volume_id": "MV-VOL-TEST", "status": "UNVERIFIED", "source_nm_to_target_voxel_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], "validation_method": "", "validation_evidence_ids": []}))
            with self.assertRaises(ValueError): build_seed_and_holdout_manifests(nodes, physical, registration, ingestion, root / "out")

    def test_import_selects_the_segmentation_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "result.npz"; output = root / "labels.npy"
            np.savez(source, segmentation=np.ones((1, 2, 3), dtype=np.uint32), probabilities=np.zeros((1, 2, 3), dtype=np.float32))
            import_ffn_labels(source, output)
            self.assertEqual(np.load(output).dtype, np.uint32)
