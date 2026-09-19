"""Read-only axis-semantics audit tests: verify that interface construction
compares the exact intended neighboring voxel for each channel_zyx, and that
the 3-member spread is geometrically symmetric across axes.

These tests assert the CURRENT implementation's semantics; they do not modify it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_g3_005_new_train_review import _members  # exact -005 geometry


def _pair_for_axis(centre, axis):
    """Reproduce the generator's pair_right rule: +1 along the affinity axis."""
    left = list(centre)
    right = left[:axis] + [left[axis] + 1] + left[axis + 1:]
    return tuple(left), tuple(right)


def test_affinity_pair_is_plus_one_along_axis_for_all_three():
    c = (10, 20, 30)
    assert _pair_for_axis(c, 0) == ((10, 20, 30), (11, 20, 30))  # Z: z+1
    assert _pair_for_axis(c, 1) == ((10, 20, 30), (10, 21, 30))  # Y: y+1
    assert _pair_for_axis(c, 2) == ((10, 20, 30), (10, 20, 31))  # X: x+1


def test_member_geometry_symmetric_across_axes():
    c = (10, 20, 30)
    for axis in (0, 1, 2):
        members = _members(c, axis)
        assert len(members) == 3
        # every member's right voxel is +1 along the affinity axis
        for left, right in members:
            assert list(right) == list(left[:axis]) + [left[axis] + 1] + list(left[axis + 1:])
        # the center member (offset 0,0) equals the centre
        assert members[0][0] == c
        # the two spread members differ from centre ONLY on the two axes
        # orthogonal to the affinity axis, by -1 and +1 respectively
        other = [d for d in range(3) if d != axis]
        spread = [members[1][0], members[2][0]]
        for m in spread:
            # affinity-axis coordinate unchanged for the left voxel
            assert m[axis] == c[axis]
            # exactly one orthogonal axis changed by +/-1
            deltas = [m[d] - c[d] for d in range(3)]
            assert deltas[axis] == 0
            assert sorted(abs(deltas[d]) for d in other) == [0, 1]


def test_z_interface_spreads_along_Y_documented_asymmetry():
    # AUDIT FINDING (documented, not a bug in the affinity pair): the 3-member
    # replicate spread uses other[0], the FIRST axis != affinity axis:
    #   Z-affinity (axis0): other=[1,2] -> spreads along Y
    #   Y-affinity (axis1): other=[0,2] -> spreads along Z
    #   X-affinity (axis2): other=[0,1] -> spreads along Z
    # The AFFINITY PAIR is always +1 along the affinity axis (symmetric); only
    # the replicate-spread direction differs across axes.
    members = _members((10, 20, 30), 0)
    lefts = [m[0] for m in members]
    assert lefts == [(10, 20, 30), (10, 19, 30), (10, 21, 30)]  # spread along Y


def test_document_actual_spread_axes_per_affinity_axis():
    # Explicitly record which axes the 3-member spread moves along, per affinity
    # axis. other = the two axes != affinity axis; offsets applied to other[0].
    expected_spread_axis = {0: 1, 1: 0, 2: 0}  # generator applies offset to other[0]
    for axis, spread_axis in expected_spread_axis.items():
        members = _members((10, 20, 30), axis)
        moved = [d for d in range(3) if members[1][0][d] != members[0][0][d]]
        assert moved == [spread_axis], f"axis {axis}: spread moved along {moved}, expected {[spread_axis]}"
