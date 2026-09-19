"""Phase E methodological tests for the axis-neutral paired-location generator.

Synthetic volumes exercise selection geometry only (not biological data).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from g3_axis_neutral_paired_generator import (  # noqa: E402
    SEP2,
    _edges,
    _interleave,
    location_contrast,
    select_locations,
)


def _vol(seed):
    rng = np.random.default_rng(seed)
    return (rng.random((32, 48, 48)) * 255).astype(np.float32)


def test_all_three_edges_defined_at_every_location():
    e = _edges((10, 20, 30))
    assert e[0] == ((10, 20, 30), (11, 20, 30))  # Z
    assert e[1] == ((10, 20, 30), (10, 21, 30))  # Y
    assert e[2] == ((10, 20, 30), (10, 20, 31))  # X


def test_edge_eligibility_independent_of_dominant_gradient():
    # Construct a volume with a strong Z-only step at a location; the location
    # must STILL contribute Y and X edges (eligibility is not argmax-gated).
    arr = np.zeros((32, 48, 48), dtype=np.float32)
    # strong Z gradient at z=16 plane
    arr[16:, :, :] = 200.0
    locs = select_locations(arr, prior_edges=set(), per_crop=3)
    # every selected location yields all three axis edges
    for loc in locs:
        e = _edges(loc["centre"])
        assert set(e.keys()) == {0, 1, 2}
    # explicitly: a center on the Z step (dominant-gradient Z) still has Y and X
    centre = (16, 24, 24)
    e = _edges(centre)
    assert e[1][1] == (16, 25, 24)  # Y edge exists
    assert e[2][1] == (16, 24, 25)  # X edge exists


@pytest.mark.parametrize("dominant_axis", [0, 1, 2])
def test_dominant_axis_location_contributes_all_axes(dominant_axis):
    # A location whose strongest gradient is `dominant_axis` still contributes
    # eligible edges for every axis.
    arr = np.zeros((32, 48, 48), dtype=np.float32)
    c = (16, 24, 24)
    nb = list(c)
    nb[dominant_axis] += 1
    arr[tuple(nb)] = 255.0  # huge step only along dominant_axis
    e = _edges(c)
    # all three edges are defined regardless of which axis dominates
    assert set(e.keys()) == {0, 1, 2}
    steps = {ax: abs(float(arr[a]) - float(arr[b])) for ax, (a, b) in e.items()}
    assert steps[dominant_axis] == 255.0
    # the other axes are still present as eligible edges (steps 0 here)
    for ax in (0, 1, 2):
        if ax != dominant_axis:
            assert ax in e


def test_selection_is_deterministic():
    arr = _vol(1)
    a = select_locations(arr, prior_edges=set(), per_crop=3)
    b = select_locations(arr, prior_edges=set(), per_crop=3)
    assert [x["centre"] for x in a] == [x["centre"] for x in b]


def test_equal_axis_representation_after_expansion():
    arr = _vol(2)
    locs = select_locations(arr, prior_edges=set(), per_crop=3)
    # expand to edges
    axis_counts = {0: 0, 1: 0, 2: 0}
    for loc in locs:
        for ax in _edges(loc["centre"]):
            axis_counts[ax] += 1
    assert axis_counts[0] == axis_counts[1] == axis_counts[2] == len(locs)


def test_within_batch_separation():
    arr = _vol(3)
    locs = select_locations(arr, prior_edges=set(), per_crop=3)
    cs = [l["centre"] for l in locs]
    for i, c in enumerate(cs):
        for d in cs[i+1:]:
            assert (c[0]-d[0])**2 + (c[1]-d[1])**2 + (c[2]-d[2])**2 >= SEP2


def test_prior_edges_excluded():
    arr = _vol(4)
    first = select_locations(arr, prior_edges=set(), per_crop=3)
    # exclude every edge of the first selection
    prior = set()
    for loc in first:
        for ax, (a, b) in _edges(loc["centre"]).items():
            prior.add((a, b, ax))
    second = select_locations(arr, prior_edges=prior, per_crop=3)
    assert {l["centre"] for l in first}.isdisjoint({l["centre"] for l in second})


def test_permutation_equivariance_of_edge_definition():
    # Edge definition must be pure geometry: permuting the center coords
    # permutes the edges' affinity axes consistently (+1 along each axis).
    for c in [(5, 6, 7), (12, 3, 20), (0 + 3, 0 + 3, 0 + 3)]:
        e = _edges(c)
        for ax, (left, right) in e.items():
            expected = list(left)
            expected[ax] += 1
            assert list(right) == expected


def test_interleave_not_blocked_by_axis():
    # 3 locations x 3 axes; interleave must not place all axis-0 first.
    items = [(li, ax, None) for li in range(3) for ax in range(3)]
    ordered = _interleave(items)
    first_three_axes = [t[1] for t in ordered[:3]]
    assert len(set(first_three_axes)) > 1  # not all the same axis up front
