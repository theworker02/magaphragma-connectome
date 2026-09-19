"""Regression coverage for portable Neuroglancer state export boundaries."""

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mvconnectome.neuroglancer import export_state


class NeuroglancerTests(unittest.TestCase):
    def test_export_has_real_sources_only_when_provided_and_source_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); nodes = root / "nodes.parquet"; edges = root / "edges.parquet"
            pq.write_table(pa.Table.from_pylist([{ "source_treenode_id": 1, "x_nm": 1.0, "y_nm": 2.0, "z_nm": 3.0 }, {"source_treenode_id": 2, "x_nm": 4.0, "y_nm": 5.0, "z_nm": 6.0}]), nodes)
            pq.write_table(pa.Table.from_pylist([{ "mv_neuron_id": "MV-N-000001", "child_treenode_id": 2, "parent_treenode_id": 1, "coordinate_frame": "MV-FRAME-CATMAID-001"}]), edges)
            output = root / "state.json"; export_state(nodes, edges, output, "precomputed://https://example.test/raw")
            state = json.loads(output.read_text()); self.assertEqual(state["layers"][0]["source"], "precomputed://https://example.test/raw")
            self.assertEqual(state["layers"][-1]["annotations"][0]["type"], "line")
