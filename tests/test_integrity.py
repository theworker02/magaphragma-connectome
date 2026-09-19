"""Regression coverage for release-integrity failures and empty registries."""

import json
import tempfile
import unittest
from pathlib import Path

from mvconnectome.integrity import verify


class IntegrityTests(unittest.TestCase):
    def _registry(self, root: Path) -> Path:
        registry = root / "registry.json"
        registry.write_text(json.dumps({"schema_version":"1.0", "sources":[{
            "id":"MV-SRC-TEST", "name":"test", "access_status":"REFERENCE", "license_status":"UNKNOWN", "landing_page":"https://example.invalid", "citation":"test", "download_approved":False
        }]}))
        return registry

    def test_empty_biological_registry_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(verify(root, self._registry(root)), [])

    def test_orphan_synapse_and_synthetic_entity_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            biological = root / "biological"
            biological.mkdir()
            (biological / "synapses.json").write_text(json.dumps({"synapses":[{
                "id":"SYN-SYN-0001", "evidence_ids":["MV-EV-404"], "dataset_ids":["MV-SRC-TEST"], "status":"MACHINE_PREDICTED", "review_state":"MACHINE_ONLY",
                "coordinates_nm_xyz":[1, 2, 3], "pre_neuron_id":"MV-N-000001", "post_neuron_id":"MV-N-000002", "segment_ids":["MV-SEG-000001"]
            }]}))
            errors = verify(root, self._registry(root))
            self.assertTrue(any("synthetic" in error for error in errors))
            self.assertTrue(any("orphan evidence" in error for error in errors))
            self.assertTrue(any("orphan neuron" in error for error in errors))
