"""Tests for the stratified Y/TRAIN sampler's selection logic.

Synthetic raw volumes exercise selection geometry only; not biological data.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from stratified_y_train_sampler import BANDS, BAND_LABELS, SEP2, select_for_crop  # noqa: E402


def _volume(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((40, 64, 64)) * 255).astype(np.float32)


def test_selection_is_deterministic(tmp_path: Path) -> None:
    arr = _volume(1)
    a = select_for_crop(arr, excluded=set(), per_band=1)
    b = select_for_crop(arr, excluded=set(), per_band=1)
    assert [x["centre_zyx"] for x in a] == [x["centre_zyx"] for x in b]
    assert [x["band"] for x in a] == [x["band"] for x in b]


def test_all_selected_are_Y_oriented(tmp_path: Path) -> None:
    arr = _volume(2)
    picks = select_for_crop(arr, excluded=set(), per_band=1)
    for pick in picks:
        for left, right, axis in pick["members"]:
            assert axis == 1
            # right is +1 on the Y axis only
            assert right[1] == left[1] + 1 and right[0] == left[0] and right[2] == left[2]


def test_bands_are_covered_when_pool_is_large(tmp_path: Path) -> None:
    arr = _volume(3)
    picks = select_for_crop(arr, excluded=set(), per_band=1)
    labels = {p["band"] for p in picks}
    # A large synthetic pool should populate all six bands.
    assert labels == set(BAND_LABELS)
    # ranks must be monotonic non-overlapping across the ordered bands
    ranks = [p["rank_within_separated"] for p in picks]
    assert ranks == sorted(ranks)


def test_separation_holds_within_batch(tmp_path: Path) -> None:
    arr = _volume(4)
    picks = select_for_crop(arr, excluded=set(), per_band=2)
    centres = [tuple(p["centre_zyx"]) for p in picks]
    for i, c in enumerate(centres):
        for d in centres[i + 1:]:
            assert (c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2 + (c[2] - d[2]) ** 2 >= SEP2


def test_excluded_pairs_are_not_reselected(tmp_path: Path) -> None:
    arr = _volume(5)
    first = select_for_crop(arr, excluded=set(), per_band=1)
    # Exclude every member pair chosen the first time.
    excluded = set()
    for p in first:
        for m in p["members"]:
            excluded.add(m)
    second = select_for_crop(arr, excluded=excluded, per_band=1)
    first_centres = {tuple(p["centre_zyx"]) for p in first}
    second_centres = {tuple(p["centre_zyx"]) for p in second}
    assert first_centres.isdisjoint(second_centres)


def test_picks_carry_raw_gradient_score(tmp_path: Path) -> None:
    # The external boundary reviewer reads item['raw_gradient_score'] for every
    # non-navigation interface question; picks must carry it or the GUI KeyErrors.
    arr = _volume(7)
    picks = select_for_crop(arr, excluded=set(), per_band=1)
    assert picks
    for p in picks:
        assert isinstance(p["raw_gradient_score"], float)


def test_selection_signature_takes_no_labels() -> None:
    # Guard: select_for_crop accepts only (arr, excluded, per_band). There is
    # no parameter through which a SAME/DIFFERENT label could enter selection.
    import inspect

    params = list(inspect.signature(select_for_crop).parameters)
    assert params == ["arr", "excluded", "per_band"]
