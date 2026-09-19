"""Permutation symmetry audit (Phase 1, H1 discriminator).

Claim under test: the interface-construction + interior + pair-reconstruction
code path is symmetric across Z/Y/X up to coordinate orientation. If we permute
which physical axis plays the 'affinity axis' role, the produced pairs and
interior verdicts must permute identically -- no axis is hard-coded special.

Uses synthetic coordinates only; read-only w.r.t. the implementation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_g3_005_new_train_review import _members


def _interior(left, channel, shape):
    """Mirror of g3_supervision._interior (axis-symmetric interior test)."""
    right = list(left)
    right[channel] += 1
    return all(0 < v < limit - 1 for v, limit in zip(left, shape)) and all(
        0 < v < limit - 1 for v, limit in zip(right, shape)
    )


def test_affinity_pair_center_member_is_permutation_equivariant():
    # The CENTER member (member 1) is the actual affinity pair that carries the
    # biological SAME/DIFFERENT decision. It MUST be permutation-equivariant:
    # permuting coordinates + axis index yields the permuted affinity pair.
    centre = (10, 20, 30)
    perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)]
    for perm in perms:
        for axis in (0, 1, 2):
            base_left, base_right = _members(centre, axis)[0]
            pcentre = tuple(centre[perm[i]] for i in range(3))
            paxis = perm.index(axis)
            perm_left, perm_right = _members(pcentre, paxis)[0]
            assert tuple(base_left[perm[i]] for i in range(3)) == perm_left, f"perm {perm} axis {axis}"
            assert tuple(base_right[perm[i]] for i in range(3)) == perm_right, f"perm {perm} axis {axis}"


def test_replicate_spread_is_NOT_permutation_equivariant_documented():
    # AUDIT FINDING: the 3-member replicate spread is NOT permutation-equivariant
    # because it always uses other[0] (first non-affinity axis). This is a known
    # geometric asymmetry in the replicate observations, NOT in the affinity pair.
    # It means Z-interfaces replicate along Y while Y- and X-interfaces replicate
    # along Z -- relevant to H3 (candidate/observation geometry), recorded here.
    spread_axis = {}
    for axis in (0, 1, 2):
        members = _members((10, 20, 30), axis)
        moved = [d for d in range(3) if members[1][0][d] != members[0][0][d]]
        spread_axis[axis] = moved[0]
    assert spread_axis == {0: 1, 1: 0, 2: 0}  # Z->Y, Y->Z, X->Z (asymmetric)


def test_interior_check_is_axis_symmetric():
    shape = (64, 128, 128)
    # A voxel one-in from the low border along each axis: interior must be
    # decided by the same rule regardless of which axis.
    for axis in (0, 1, 2):
        # left at coordinate 1 on the affinity axis -> 0 < 1 < limit-1 True; right=2 True
        left = [10, 10, 10]
        left[axis] = 1
        assert _interior(tuple(left), axis, shape) is True
        # left at 0 on the affinity axis -> 0 < 0 is False -> not interior
        left2 = [10, 10, 10]
        left2[axis] = 0
        assert _interior(tuple(left2), axis, shape) is False
        # right at the far border
        left3 = [10, 10, 10]
        left3[axis] = shape[axis] - 2  # right = limit-1 -> 0<limit-1<limit-1 False
        assert _interior(tuple(left3), axis, shape) is False


def test_no_axis_is_numerically_privileged_in_pair_offset():
    # The +1 affinity offset magnitude is identical for all axes.
    for axis in (0, 1, 2):
        (l, r) = _members((10, 20, 30), axis)[0]
        assert r[axis] - l[axis] == 1
        assert sum(abs(r[d] - l[d]) for d in range(3)) == 1
