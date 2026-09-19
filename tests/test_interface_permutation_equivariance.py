"""Axis permutation-equivariance audit for the G3 interface construction.

Read-only. Reproduces the EXACT current source geometry (from the generator /
-005 builder) and tests two independent properties:

  A. AFFINITY-EDGE equivariance: the canonical center pair (left,right,axis)
     inverse-maps to the original physical adjacent edge under every axis
     permutation.
  B. FULL-INTERFACE equivariance: the whole 3-member interface (as a set of
     physical edges) is invariant under permute -> regenerate -> inverse-map.

If A passes and B fails, H1a (pair/coord bug) is NOT supported and
H1b (interface-context geometric asymmetry) IS supported.

The current construction is reproduced verbatim below as CURRENT_members().
"""
import itertools
import sys
from pathlib import Path

# ---- Verbatim reproduction of the current source construction --------------
# generator: other=[d for d in range(3) if d!=axis]; offsets=((0,0),(-1,0),(1,0));
#   left[other[0]]+=offset[0]; left[other[1]]+=offset[1]
# pair_right = left[:axis] + [left[axis]+1] + left[axis+1:]
def CURRENT_members(centre, axis):
    other = [d for d in range(3) if d != axis]
    offsets = ((0, 0), (-1, 0), (1, 0))
    out = []
    for off in offsets:
        left = list(centre)
        left[other[0]] += off[0]
        left[other[1]] += off[1]
        right = left[:axis] + [left[axis] + 1] + left[axis + 1:]
        out.append((tuple(left), tuple(right), axis))
    return out


def center_pair(centre, axis):
    left = list(centre)
    right = left[:axis] + [left[axis] + 1] + left[axis + 1:]
    return (tuple(left), tuple(right), axis)


# ---- Permutation machinery -------------------------------------------------
PERMS = list(itertools.permutations((0, 1, 2)))  # perm p maps original axis a -> position p[a]


def permute_coord(c, perm):
    # new coordinate: axis 'perm[a]' receives original component c[a]
    out = [0, 0, 0]
    for a in range(3):
        out[perm[a]] = c[a]
    return tuple(out)


def inverse_perm(perm):
    inv = [0, 0, 0]
    for a in range(3):
        inv[perm[a]] = a
    return tuple(inv)


def canonical_edge(left, right, axis):
    """A physical undirected edge as a frozenset of the two voxels."""
    return frozenset({tuple(left), tuple(right)})


def test_affinity_edge_equivariance_all_perms():
    """Property A: center edge inverse-maps to the original edge for all perms."""
    centre = (10, 20, 30)
    results = {}
    for axis in (0, 1, 2):
        orig_left, orig_right, _ = center_pair(centre, axis)
        orig_edge = canonical_edge(orig_left, orig_right, axis)
        for perm in PERMS:
            pcentre = permute_coord(centre, perm)
            paxis = perm[axis]
            pl, pr, pa = center_pair(pcentre, paxis)
            inv = inverse_perm(perm)
            back_left = permute_coord(pl, inv)
            back_right = permute_coord(pr, inv)
            back_edge = canonical_edge(back_left, back_right, axis)
            results[(axis, perm)] = (back_edge == orig_edge)
    assert all(results.values()), [k for k, v in results.items() if not v]


def build_full_interface_failure_table():
    """Property B: full 3-member interface equivariance. Returns failure rows."""
    centre = (10, 20, 30)
    failures = []
    for axis in (0, 1, 2):
        orig = CURRENT_members(centre, axis)
        orig_edges = {canonical_edge(*m) for m in orig}
        orig_offsets = [tuple(l[d] - centre[d] for d in range(3)) for (l, _, _) in orig]
        for perm in PERMS:
            pcentre = permute_coord(centre, perm)
            paxis = perm[axis]
            pmembers = CURRENT_members(pcentre, paxis)
            inv = inverse_perm(perm)
            back_edges = {canonical_edge(permute_coord(l, inv), permute_coord(r, inv), axis) for (l, r, _) in pmembers}
            if back_edges != orig_edges:
                # which orthogonal direction changed
                back_offsets = sorted(
                    tuple(permute_coord(l, inv)[d] - centre[d] for d in range(3)) for (l, _, _) in pmembers
                )
                center_same = center_pair(centre, axis)[:2] == (
                    permute_coord(pmembers[0][0], inv), permute_coord(pmembers[0][1], inv))
                failures.append({
                    "original_affinity_axis": axis,
                    "permutation_orig_to_pos": perm,
                    "original_member_offsets": sorted(orig_offsets),
                    "inverse_mapped_member_offsets": back_offsets,
                    "center_edge_preserved": center_same,
                    "only_replicate_members_changed": center_same,
                })
    return failures


def test_full_interface_equivariance_fails_and_records_table(capsys):
    """Property B is EXPECTED TO FAIL for the current one-orthogonal-axis spread.
    We assert the failure is confined to replicate members (center edge intact),
    which is the precise signature of H1b (not H1a)."""
    failures = build_full_interface_failure_table()
    # There must be failures (otherwise B would be equivariant and H1b unsupported).
    assert failures, "expected full-interface equivariance to fail for current spread"
    # Every failure must preserve the center edge (H1a not supported).
    assert all(f["center_edge_preserved"] for f in failures), failures
    assert all(f["only_replicate_members_changed"] for f in failures), failures


def test_spread_axis_is_other0_source_signature():
    """Source-level: spread_axis == other[0] (first axis != affinity axis)."""
    table = {}
    for axis in (0, 1, 2):
        members = CURRENT_members((10, 20, 30), axis)
        moved = sorted({d for m in members for d in range(3) if m[0][d] != 10 * (d == 0) + 20 * (d == 1) + 30 * (d == 2)})
        table[axis] = moved
    # Z(0) spreads along Y(1); Y(1) spreads along Z(0); X(2) spreads along Z(0)
    assert table[0] == [1]
    assert table[1] == [0]
    assert table[2] == [0]
