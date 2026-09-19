"""Compare the FIXED sampler's canonical selection against the already-
materialized -008 queues on disk. Read-only; writes nothing.

Validates whether the existing (unreviewed) -008 package is exactly what the
corrected, deterministic sampler produces. If identical, the existing package
can be validated in place rather than superseded."""
from __future__ import annotations

import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO / "tools"))
from build_g3_008_axis_neutral import select_only, _edges, _axis_index  # noqa

sel = select_only()

# Expected edges from the fixed selection: 12 centers x 3 axis edges.
expected = set()
expected_centers = []
for it in sel["selection"]:
    c = tuple(it["center_zyx"])
    expected_centers.append(c)
    e = _edges(c)
    for ax in ("Z", "Y", "X"):
        expected.add((tuple(e[ax][0]), tuple(e[ax][1]), _axis_index(ax)))

# Materialized edges from the existing queues on disk.
materialized = set()
mat_centers = set()
qfiles = sorted(glob.glob(str(REPO / "experiments/phase6e/g3-interface-queues-008/MV-G3-AXNEU4-*.json")))
for qp in qfiles:
    q = json.loads(Path(qp).read_text())
    for question in q["questions"]:
        materialized.add((tuple(question["pair_left_zyx"]), tuple(question["pair_right_zyx"]), int(question["channel_zyx"])))
    # center = the Z-edge left voxel (member 1)
    for question in q["questions"]:
        if question["axis_name"] == "Z":
            mat_centers.add(tuple(question["pair_left_zyx"]))

report = {
    "existing_queue_files": len(qfiles),
    "expected_center_count": len(expected_centers),
    "materialized_center_count": len(mat_centers),
    "centers_identical": set(expected_centers) == mat_centers,
    "expected_edge_count": len(expected),
    "materialized_edge_count": len(materialized),
    "edges_identical": expected == materialized,
    "edges_only_in_fixed_selection": sorted([list(l), list(r), c] for (l, r, c) in list(expected - materialized)[:10]),
    "edges_only_on_disk": sorted([list(l), list(r), c] for (l, r, c) in list(materialized - expected)[:10]),
}
print(json.dumps(report, indent=2))
