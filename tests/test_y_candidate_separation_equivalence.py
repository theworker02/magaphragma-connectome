"""Equivalence test: spatial-hash greedy separation must produce the exact
same accepted centres in the same order as the brute-force reference.

This guards the diagnostic's only performance-motivated change. Coordinates
here are synthetic geometry, not biological data.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from diagnose_y_train_candidate_pool import (  # noqa: E402
    SEP2,
    greedy_separated_bruteforce,
    greedy_separated_hashed,
)


def _ranked_coords(seed: int, n: int, extent: int) -> np.ndarray:
    """A deterministic ranked candidate list (rank == array order)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, extent, size=(n, 3))


@pytest.mark.parametrize("seed", range(8))
def test_hashed_matches_bruteforce_dense(seed: int) -> None:
    # Small extent + many points forces many separation conflicts.
    coords = _ranked_coords(seed, n=300, extent=30)
    assert greedy_separated_hashed(coords) == greedy_separated_bruteforce(coords)


@pytest.mark.parametrize("seed", range(4))
def test_hashed_matches_bruteforce_sparse(seed: int) -> None:
    # Large extent -> few conflicts, most accepted.
    coords = _ranked_coords(seed, n=200, extent=200)
    assert greedy_separated_hashed(coords) == greedy_separated_bruteforce(coords)


def test_exact_boundary_distance_predicate() -> None:
    # Points exactly at distance^2 == SEP2 must be ACCEPTED (predicate is < SEP2
    # reject, i.e. >= SEP2 accept), matching the generator.
    sep = int(round(SEP2 ** 0.5))
    coords = np.array([[0, 0, 0], [0, 0, sep]])  # distance exactly sep
    hashed = greedy_separated_hashed(coords)
    brute = greedy_separated_bruteforce(coords)
    assert hashed == brute == [(0, 0, 0), (0, 0, sep)]


def test_just_inside_radius_is_rejected() -> None:
    sep = int(round(SEP2 ** 0.5))
    coords = np.array([[0, 0, 0], [0, 0, sep - 1]])  # closer than sep -> reject 2nd
    assert greedy_separated_hashed(coords) == greedy_separated_bruteforce(coords) == [(0, 0, 0)]


def test_order_is_preserved() -> None:
    coords = _ranked_coords(123, n=150, extent=40)
    result = greedy_separated_hashed(coords)
    # Accepted centres must appear in the same relative order they were walked.
    walked_order = [tuple(int(v) for v in c) for c in coords]
    positions = [walked_order.index(c) for c in result]
    assert positions == sorted(positions)
