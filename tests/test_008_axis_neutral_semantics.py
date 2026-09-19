"""Adversarial synthetic equivalence + determinism suite for the -008
deterministic axis-neutral sampler.

Proves the vectorized/float64 implementation matches a pure-python reference
EXACTLY (ordered eligible coords, contrast values with atol=0, separated
centers, pure-python int center repr, hash-rank pick) across adversarial cases:
tiny/boundary shapes, constant volumes, tied contrasts, quantile-boundary
gradients, multiple candidates within the separation radius, odd/even shapes,
and seeded random volumes. Also proves selection determinism.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_g3_008_axis_neutral as B


# ---- pure-python reference (float64) --------------------------------------
def ref_eligible(arr):
    a = arr.astype(np.float64)
    z, y, x = a.shape
    coords, cvals = [], []
    for zi in range(B.MARGIN, z - B.MARGIN - 1):
        for yi in range(B.MARGIN, y - B.MARGIN - 1):
            for xi in range(B.MARGIN, x - B.MARGIN - 1):
                c0 = float(a[zi, yi, xi])
                ez = abs(float(a[zi + 1, yi, xi]) - c0)
                ey = abs(float(a[zi, yi + 1, xi]) - c0)
                ex = abs(float(a[zi, yi, xi + 1]) - c0)
                coords.append((zi, yi, xi))
                cvals.append((ez + ey + ex) / 3.0)
    return coords, cvals


def ref_greedy(coords, cvals):
    sep = 12
    grid, kept = {}, []
    for c, ct in zip(coords, cvals):
        z, y, x = c
        cell = (z // sep, y // sep, x // sep)
        conflict = False
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for (az, ay, ax) in grid.get((cell[0] + dz, cell[1] + dy, cell[2] + dx), ()):
                        if (az - z) ** 2 + (ay - y) ** 2 + (ax - x) ** 2 < B.SEP2:
                            conflict = True
                            break
                    if conflict: break
                if conflict: break
            if conflict: break
        if not conflict:
            grid.setdefault(cell, []).append((z, y, x))
            kept.append((c, ct))
    return kept


# ---- adversarial volume factory -------------------------------------------
def _cases():
    cases = {}
    # tiny volumes at/above the minimum interior size (need > 2*MARGIN+1)
    cases["tiny_min"] = np.zeros((9, 9, 9), np.float32)
    # constant volume -> all contrasts identical (ties everywhere)
    cases["constant"] = np.full((12, 12, 12), 128, np.float32)
    # repeated/tied contrast values via a low-bit-depth pattern
    r = np.random.default_rng(0)
    cases["tied_lowbits"] = (r.integers(0, 3, (14, 15, 16)) * 40).astype(np.float32)
    # gradients placed to hit quantile boundaries: block structure
    v = np.zeros((16, 16, 16), np.float32)
    v[:8] = 10; v[8:] = 50
    cases["block_boundary"] = v
    # multiple candidates within separation radius: dense high-contrast cluster
    v2 = np.zeros((20, 20, 20), np.float32)
    v2[10, 10, 10] = 200; v2[10, 10, 12] = 200; v2[10, 12, 10] = 200
    cases["dense_cluster"] = v2
    # odd/even shapes
    cases["odd_even"] = (np.random.default_rng(7).random((13, 18, 15)) * 255).astype(np.float32)
    # seeded random volumes
    for s in range(4):
        cases[f"rand_{s}"] = (np.random.default_rng(100 + s).random((20, 22, 24)) * 255).astype(np.float32)
    return cases


CASES = _cases()


def test_eligible_coords_and_contrast_exact():
    for name, arr in CASES.items():
        oc, ocv = B.eligible_centers(arr)
        rc, rcv = ref_eligible(arr)
        assert oc == rc, f"{name}: coord order/list differs"
        # exact equality (atol=0) of contrast values
        assert ocv == rcv, f"{name}: contrast values differ (not exact)"
        # pure-python types
        assert all(type(v) is int for c in oc for v in c), f"{name}: non-int coords"
        assert all(type(v) is float for v in ocv), f"{name}: non-float contrasts"


def test_separated_centers_exact():
    for name, arr in CASES.items():
        oc, ocv = B.eligible_centers(arr)
        o_sep = B.greedy_separated_centers(oc, ocv)
        r_sep = ref_greedy(*ref_eligible(arr))
        assert [c for c, _ in o_sep] == [c for c, _ in r_sep], f"{name}: separated centers differ"
        assert [ct for _, ct in o_sep] == [ct for _, ct in r_sep], f"{name}: separated contrasts differ"


def test_center_repr_is_pure_python_int():
    arr = CASES["rand_0"]
    o_sep = B.greedy_separated_centers(*B.eligible_centers(arr))
    center = o_sep[0][0]
    assert str(center) == f"({center[0]}, {center[1]}, {center[2]})"
    assert all(type(v) is int for v in center)


def test_hash_rank_pick_matches_reference():
    for name in ("dense_cluster", "rand_1", "tied_lowbits"):
        arr = CASES[name]
        o_sep = B.greedy_separated_centers(*B.eligible_centers(arr))
        r_sep = ref_greedy(*ref_eligible(arr))
        if not o_sep:
            continue
        key = lambda cc: B._hash_rank("sha", "SRC", "Q1", str(cc[0]))
        assert tuple(min(o_sep, key=key)[0]) == tuple(min(r_sep, key=key)[0]), name


def test_edges_are_plus_one_per_axis():
    e = B._edges((10, 20, 30))
    assert e["Z"] == [[10, 20, 30], [11, 20, 30]]
    assert e["Y"] == [[10, 20, 30], [10, 21, 30]]
    assert e["X"] == [[10, 20, 30], [10, 20, 31]]


# ---- structural gate + on-disk provenance (real data) --------------------
def test_build_invariants_pass_on_real_selection():
    """_assert_build_invariants must accept the canonical real selection."""
    sel = B.select_only()
    # Must not raise.
    B._assert_build_invariants(sel)
    assert len(sel["selection"]) == B.N_CROPS
    strata = [it["stratum"] for it in sel["selection"]]
    assert {s: strata.count(s) for s in B.STRATA} == {s: 3 for s in B.STRATA}


def test_pinned_snapshot_stable_and_713():
    sel = B.select_only()
    assert sel["pinned_exclusion_count"] == 713
    # recomputing the live snapshot must reproduce the same hash (queues-008 and
    # packages-008 are skipped, so building -008 does not perturb it)
    _, h = B.pinned_historical_edges()
    assert h == sel["pinned_exclusion_sha256"]


def test_onchain_008_matches_canonical_and_edges_excluded():
    import glob
    import json as _json
    from pathlib import Path as _P

    prot = _json.loads((B.REPO / "experiments/phase6e/G3_AXIS_NEUTRAL_SAMPLING_PROTOCOL_004.json").read_text())
    hist = {(tuple(l), tuple(r), c) for (l, r, c) in prot["pinned_exclusion_edges"]}
    qfiles = sorted(glob.glob(str(B.Q_ROOT / "MV-G3-AXNEU4-*.json")))
    assert len(qfiles) == 12
    axes = {"Z": 0, "Y": 0, "X": 0}
    centers = []
    for qf in qfiles:
        q = _json.loads(_P(qf).read_text())
        assert q["status"] == "EXPERT_INTERFACE_REVIEW_REQUIRED"
        qq = q["questions"]
        assert len(qq) == 3
        for e in qq:
            axes[e["axis_name"]] += 1
            key = (tuple(e["pair_left_zyx"]), tuple(e["pair_right_zyx"]), int(e["channel_zyx"]))
            assert key not in hist, f"on-disk edge {key} collides with pinned snapshot"
        # 3 edges share one physical center (paired-location)
        lefts = {tuple(e["pair_left_zyx"]) for e in qq}
        assert len(lefts) == 1
        centers.append(next(iter(lefts)))
    assert axes == {"Z": 12, "Y": 12, "X": 12}
    # 12 distinct physical centers
    assert len(set(centers)) == 12
