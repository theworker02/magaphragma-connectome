"""Unit tests for half-open source bounds and deterministic work chunking."""

import unittest

from mvconnectome.production_domain import _bounds_from_info, _chunk_index


class ProductionDomainTests(unittest.TestCase):
    def test_parent_bounds_are_half_open_and_chunk_neighbors_are_stable(self):
        bounds = _bounds_from_info({"Extended": {"MinPoint": [0, 0, 0], "MaxPoint": [2047, 1023, 255]}})
        self.assertEqual(bounds, {"x": [0, 2048], "y": [0, 1024], "z": [0, 256]})
        chunks = _chunk_index(bounds, (1024, 1024, 128), (64, 64, 16))
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0]["id"], "MV-CHUNK-10000001")
        self.assertIn("MV-CHUNK-10000002", chunks[0]["neighbor_chunk_ids"])
        self.assertEqual(chunks[-1]["core_bounds_xyz"]["z"], [128, 256])
