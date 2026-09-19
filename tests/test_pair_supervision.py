"""Unit test for adjacent affinity-pair extraction without array wraparound."""

import numpy as np

from mvconnectome.pair_supervision import OFFSETS_ZYX, _slices


def test_offsets_cover_expected_neighbor_pairs_without_wraparound():
    shape = (3, 4, 5)
    for offset in OFFSETS_ZYX:
        left, right = _slices(shape, offset)
        first = np.zeros(shape, dtype=np.uint8)[left]
        second = np.zeros(shape, dtype=np.uint8)[right]
        assert first.shape == second.shape
        assert all(size == limit - 1 for size, limit, delta in zip(first.shape, shape, offset) if delta)
