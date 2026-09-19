"""Regression coverage for honest provenance reporting from empty registries."""

import json
import tempfile
import unittest
from pathlib import Path

from mvconnectome.reporting import provenance_report


class ReportingTests(unittest.TestCase):
    def test_empty_registry_report_is_explicit_not_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry.json"
            registry.write_text(json.dumps({"schema_version":"1.0", "sources":[{
                "id":"MV-SRC-TEST", "doi":"10.example/test", "license_status":"UNKNOWN", "access_status":"REFERENCE", "citation":"test"
            }]}))
            output = provenance_report(root, registry, root / "report.json")
            report = json.loads(output.read_text())
            self.assertEqual(report["biological_registry"]["dataset_state"], "EMPTY")
            self.assertEqual(report["biological_registry"]["counts"]["neurons"], 0)
