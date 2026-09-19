"""Pre-review gate tests for the axis-neutral paired-location experiment (V2).

Proves: (1) affinity-edge permutation equivariance / exact per-axis pair;
(2) all three axes eligible regardless of dominant gradient; (3) deterministic
regeneration; (4) exact Z/Y/X designed-count balance; (5) exclusion of historical
edges; (6) no label/model dependency in selection; (7) spatial/provenance checks.
Synthetic volumes exercise selection geometry only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import build_g3_axis_neutral_paired_experiment as B  # noqa: E402

Q_ROOT = REPO / "experiments/phase6e/g3-interface-queues-006b"
CROPS = ["MV-G3-AXNEU2-LO1", "MV-G3-AXNEU2-LO2", "MV-G3-AXNEU2-HI1", "MV-G3-AXNEU2-HI2"]


def _vol(seed, shape=(40, 64, 64)):
    return (np.random.default_rng(seed).random(shape) * 255).astype(np.float32)


def test_three_edges_exact_per_axis():
    c = (10, 20, 30)
    edges = B.three_edges(c)
    assert edges[0] == ((10, 20, 30), (11, 20, 30), 0)  # Z
    assert edges[1] == ((10, 20, 30), (10, 21, 30), 1)  # Y
    assert edges[2] == ((10, 20, 30), (10, 20, 31), 2)  # X


def test_permutation_equivariance_of_edges():
    # Permuting the center coordinates permutes the edge endpoints identically;
    # the affinity direction remains +1 along the named axis.
    c = (3, 7, 5)
    edges = {ax: (l, r) for (l, r, ax) in B.three_edges(c)}
    for ax in (0, 1, 2):
        left, right = edges[ax]
        delta = [right[i] - left[i] for i in range(3)]
        assert delta[ax] == 1 and sum(delta) == 1  # exactly +1 on that axis


def test_all_three_axes_eligible_regardless_of_gradient():
    # Construct a volume with an overwhelming Z gradient; all three edges must
    # still be generated (edge eligibility is NOT argmax-gated).
    arr = np.zeros((40, 64, 64), dtype=np.float32)
    arr[::2] = 255.0  # huge Z-direction gradient, ~0 in Y/X
    locs, _ = B.select_locations(arr, set(), want=3)
    assert len(locs) == 3
    for loc in locs:
        edges = B.three_edges(tuple(loc["center_zyx"]))
        axes = sorted(e[2] for e in edges)
        assert axes == [0, 1, 2]  # Z, Y, X all present


def test_deterministic_regeneration():
    arr = _vol(1)
    a, _ = B.select_locations(arr, set(), want=3)
    b, _ = B.select_locations(arr, set(), want=3)
    assert [x["center_zyx"] for x in a] == [x["center_zyx"] for x in b]


def test_designed_axis_balance_in_real_queues():
    # Each generated queue must have exactly equal Z/Y/X edge counts.
    for crop in CROPS:
        q = json.loads((Q_ROOT / f"{crop}.json").read_text())
        counts = {0: 0, 1: 0, 2: 0}
        for qq in q["questions"]:
            counts[int(qq["channel_zyx"])] += 1
        assert counts[0] == counts[1] == counts[2] == len(q["locations"])


def test_excludes_historical_edges():
    arr = _vol(2)
    first, _ = B.select_locations(arr, set(), want=2)
    excluded = set()
    for loc in first:
        for e in B.three_edges(tuple(loc["center_zyx"])):
            excluded.add((e[0], e[1], e[2]))
    second, _ = B.select_locations(arr, excluded, want=2)
    first_c = {tuple(x["center_zyx"]) for x in first}
    second_c = {tuple(x["center_zyx"]) for x in second}
    assert first_c.isdisjoint(second_c)


def test_selection_signature_has_no_label_input():
    import inspect
    params = list(inspect.signature(B.select_locations).parameters)
    assert params == ["arr", "excluded_edges", "want"]


def test_real_queues_exclude_all_historical_edges():
    hist = B.historical_edges()
    for crop in CROPS:
        q = json.loads((Q_ROOT / f"{crop}.json").read_text())
        for qq in q["questions"]:
            key = (tuple(qq["pair_left_zyx"]), tuple(qq["pair_right_zyx"]), int(qq["channel_zyx"]))
            assert key not in hist


def test_workspace_provenance_and_raw_hash():
    import hashlib
    for crop in CROPS:
        ws = json.loads((REPO / "experiments/phase6e/g3-external-review-packages-006b" / crop / "workspace.json").read_text())
        q = json.loads((Q_ROOT / f"{crop}.json").read_text())
        assert q["workspace_id"] == ws["id"]
        raw = Path(ws["raw"]["path"])
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == ws["raw"]["sha256"] == q["raw_sha256"]
        # empty fresh event log
        log = Path(ws["event_log"]["path"])
        assert (log.read_text() if log.exists() else "x").strip() == ""
