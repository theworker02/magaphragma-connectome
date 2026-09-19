"""Regression coverage for evidence joins exposed to local API consumers."""

import json
import tempfile
import unittest
from pathlib import Path

from mvconnectome.repository import ConnectomeRepository


class RepositoryTests(unittest.TestCase):
    def test_source_dataset_is_exposed_without_inventing_biology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "datasets").mkdir()
            (root / "datasets" / "registry.json").write_text(json.dumps({"sources": [{
                "id": "MV-SRC-VERIFIED", "citation": "source citation", "access_status": "REFERENCE"
            }]}))
            repository = ConnectomeRepository(root)
            source = repository.source_dataset("MV-SRC-VERIFIED")
            self.assertEqual(source["citation"], "source citation")
            self.assertEqual(repository.status()["counts"]["neurons"], 0)
