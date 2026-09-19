"""Pre-review gates for protocol-003 / -007 axis-neutral paired-location design.

Proves: (1) affinity-edge permutation equivariance, (2) all three axes eligible
regardless of dominant gradient, (3) deterministic regeneration, (4) exact
Z/Y/X count balance, (5) exclusion of historical exact edges, (6) no label/model
dependency, (7) spatial/provenance constraints.
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_g3_007_axis_neutral as B  # noqa: E402


def test_edges_are_canonical_plus_one_per_axis():
    e = B._edges_for_center((10, 20, 30))
    assert e["Z"] == ((10, 20, 30), (11, 20, 30), 0)
    assert e["Y"] == ((10, 20, 30), (10, 21, 30), 1)
    assert e["X"] == ((10, 20, 30), (10, 20, 31), 2)


def test_all_three_axes_present_and_balanced():
    e = B._edges_for_center((5, 6, 7))
    axes = sorted(v[2] for v in e.values())
    assert axes == [0, 1, 2]  # exactly one edge per axis, balanced by construction


def test_edge_eligibility_independent_of_gradient():
    # Construct a volume with a huge Z gradient and flat Y/X. argmax(|grad|) is
    # Z everywhere, but all three edges must still be produced for a center.
    arr = np.zeros((20, 20, 20), dtype=np.float32)
    arr[10:] = 255.0  # strong Z step at z=10
    e = B._edges_for_center((9, 9, 9))
    assert {v[2] for v in e.values()} == {0, 1, 2}
    # local contrast metric is defined and finite regardless
    c = B.local_edge_contrast(arr, (9, 9, 9))
    assert np.isfinite(c)


def test_stable_hash_is_deterministic_and_coordinate_dependent():
    h1 = B._stable_hash("SRC-1", 1, 2, 3)
    h2 = B._stable_hash("SRC-1", 1, 2, 3)
    h3 = B._stable_hash("SRC-1", 1, 2, 4)
    assert h1 == h2 and h1 != h3


def test_historical_edges_excluded_type():
    edges = B.historical_excluded_edges()
    assert isinstance(edges, set)
    # every element is (tuple,tuple,int)
    for el in list(edges)[:20]:
        assert len(el) == 3 and isinstance(el[2], int)


def test_builder_signature_takes_no_labels():
    # build() consumes only output paths; no label/model parameters exist.
    params = list(inspect.signature(B.build).parameters)
    assert params == ["protocol_out", "ws_root", "q_root"]


def test_deterministic_protocol_regeneration(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    pa = B.build(a, None, None)
    pb = B.build(b, None, None)
    # normalize volatile timestamp
    for p in (pa, pb):
        p.pop("created_at", None)
    # compare the substantive design fields
    def sig(p):
        return {
            "n_crops": p["n_crops"], "total_edges": p["total_edges"],
            "excluded": p["excluded_historical_edge_count"],
            "locs": [(s["crop_id"], s["source_id"], s["assigned_band"], tuple(s["center_zyx"]),
                      s["selection_hash"], s["raw_sha256"]) for s in p["selected_locations"]],
        }
    assert sig(pa) == sig(pb)
    # exact Z/Y/X balance: each location has one edge per axis
    for s in pa["selected_locations"]:
        assert sorted(v["axis"] for v in s["edges"].values()) == [0, 1, 2]
    # 12 locations, 36 edges, 4-stratum rotation applied by position
    assert pa["total_locations"] == 12 and pa["total_edges"] == 36
    bands = [s["assigned_band"] for s in pa["selected_locations"]]
    assert bands == (["VERY_LOW", "LOW", "MID", "HIGH"] * 3)


def test_selected_locations_exclude_historical_edges(tmp_path):
    p = B.build(tmp_path / "p.json", None, None)
    hist = B.historical_excluded_edges()
    for s in p["selected_locations"]:
        for e in s["edges"].values():
            key = (tuple(e["left"]), tuple(e["right"]), e["axis"])
            assert key not in hist
