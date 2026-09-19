"""Semantics-preservation gate for the -007 selection pipeline.

Proves, on small synthetic volumes (fast, exhaustive), that a PURE-PYTHON
REFERENCE implementation and the OPTIMIZED (vectorized) implementation in
build_g3_007_axis_neutral_paired produce byte-identical:
  * eligible center ordered coordinate lists
  * contrast values
  * separated-center ordered lists
  * the deterministic hash-rank pick (incl. str() representation of the center)

The str()-representation check is the key trap: if the optimized path leaves
NumPy int64 in the center tuple, str(center) differs from the pure-python
"(z, y, x)" and the sha256 hash-rank pick diverges silently.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_g3_007_axis_neutral_paired as B

MARGIN = B.MARGIN
SEP2 = B.SEP2


def ref_eligible_centers(arr):
    z, y, x = arr.shape
    coords, contrasts = [], []
    for zi in range(MARGIN, z - MARGIN - 1):
        for yi in range(MARGIN, y - MARGIN - 1):
            for xi in range(MARGIN, x - MARGIN - 1):
                c0 = float(arr[zi, yi, xi])
                ez = abs(float(arr[zi + 1, yi, xi]) - c0)
                ey = abs(float(arr[zi, yi + 1, xi]) - c0)
                ex = abs(float(arr[zi, yi, xi + 1]) - c0)
                coords.append((zi, yi, xi))
                contrasts.append((ez + ey + ex) / 3.0)
    return coords, contrasts


def ref_greedy(coords, contrasts):
    sep = 12
    grid, kept = {}, []
    for c, ct in zip(coords, contrasts):
        z, y, x = c
        cell = (z // sep, y // sep, x // sep)
        conflict = False
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for (az, ay, ax) in grid.get((cell[0] + dz, cell[1] + dy, cell[2] + dx), ()):
                        if (az - z) ** 2 + (ay - y) ** 2 + (ax - x) ** 2 < SEP2:
                            conflict = True
                            break
                    if conflict: break
                if conflict: break
            if conflict: break
        if not conflict:
            grid.setdefault(cell, []).append((z, y, x))
            kept.append((c, ct))
    return kept


def _vol(seed, shape=(20, 24, 24)):
    rng = np.random.default_rng(seed)
    return (rng.random(shape) * 255).astype(np.float32)


def test_eligible_centers_match():
    for seed in range(4):
        arr = _vol(seed)
        o_coords, o_contrast = B.eligible_centers(arr)
        r_coords, r_contrast = ref_eligible_centers(arr)
        assert [tuple(int(v) for v in c) for c in o_coords] == r_coords
        assert np.allclose([float(v) for v in o_contrast], r_contrast, atol=0, rtol=0)


def test_separated_centers_match():
    for seed in range(4):
        arr = _vol(seed)
        o_sep = B.greedy_separated_centers(B.eligible_centers(arr))
        r_coords, r_contrast = ref_eligible_centers(arr)
        r_sep = ref_greedy(r_coords, r_contrast)
        assert [tuple(int(v) for v in c) for c, _ in o_sep] == [tuple(c) for c, _ in r_sep]


def test_center_repr_is_pure_python_int_not_numpy():
    # The hash-rank pick uses str(center). Guard against NumPy-int repr trap.
    arr = _vol(1)
    o_sep = B.greedy_separated_centers(B.eligible_centers(arr))
    center = o_sep[0][0]
    assert all(type(v) is int for v in center), f"center holds non-int types: {[type(v) for v in center]}"
    assert str(center) == f"({center[0]}, {center[1]}, {center[2]})"


def test_hash_rank_pick_matches_reference():
    # Full pick pipeline on one synthetic crop, single stratum, no history.
    arr = _vol(2)
    o_sep = B.greedy_separated_centers(B.eligible_centers(arr))
    r_sep = ref_greedy(*ref_eligible_centers(arr))
    raw_sha, sid, stratum = "deadbeef", "MV-SYN", "Q1"
    o_pick = min(o_sep, key=lambda cc: B._hash_rank(raw_sha, sid, stratum, str(cc[0])))
    r_pick = min(r_sep, key=lambda cc: B._hash_rank(raw_sha, sid, stratum, str(cc[0])))
    assert tuple(int(v) for v in o_pick[0]) == tuple(r_pick[0])
