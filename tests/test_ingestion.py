"""Regression coverage for checksum- and metadata-gated volume ingestion."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mvconnectome.ingestion import ingest_volume


class IngestionTests(unittest.TestCase):
    def test_ingestion_requires_authoritative_hash_and_writes_registration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volume = root / "sample.tiff"
            volume.write_bytes(b"real-but-tiny-test-artifact")
            digest = hashlib.sha256(volume.read_bytes()).hexdigest()
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({
                "dataset_id": "MV-SRC-WASPSYN-2024", "specimen_id": "SOURCE-SPECIMEN-UNKNOWN", "format": "tiff",
                "voxel_size_nm": [8, 8, 8], "shape_zyx": [2, 2, 2], "coordinate_frame": "source-zyx",
                "origin_nm_xyz": [0, 0, 0], "source_url": "https://example.invalid/artifact", "source_sha256": digest,
                "evidence_status": "SOURCE_ANNOTATED"
            }))
            result = ingest_volume(volume, metadata, root / "datasets")
            self.assertTrue(result.exists())
            self.assertIn(digest, result.read_text())

    def test_ingestion_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volume = root / "sample.tiff"
            volume.write_bytes(b"artifact")
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({
                "dataset_id": "MV-SRC-WASPSYN-2024", "specimen_id": "x", "format": "tiff", "voxel_size_nm": [8,8,8],
                "shape_zyx": [2,2,2], "coordinate_frame": "source", "origin_nm_xyz": [0,0,0], "source_url": "x",
                "source_sha256": "0" * 64, "evidence_status": "SOURCE_ANNOTATED"
            }))
            with self.assertRaises(ValueError):
                ingest_volume(volume, metadata, root / "datasets")
