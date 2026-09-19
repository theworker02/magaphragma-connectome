"""Read-only: verify the frozen pinned-exclusion snapshot's per_source.avoid_edges,
unioned + canonicalized, equals EXACTLY the live-glob edge set the current
sampler reproduces (count 713, sha 2cbabb99...). If equal, switching the sampler
to load the snapshot changes no scientific selection input."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SNAP = REPO / "experiments/phase6e/G3_008_PINNED_EXCLUSION_SNAPSHOT_001.json"

# import the live-glob function from the current sampler
import sys
sys.path.insert(0, str(REPO / "tools"))
from build_g3_008_axis_neutral import pinned_historical_edges  # noqa

snap = json.loads(SNAP.read_text())

# Union all avoid_edges from per_source, canonicalize to sorted [ [l],[r],c ].
snap_edges = set()
for src in snap.get("per_source", []):
    for edge in src.get("avoid_edges", []):
        left, right, channel = edge[0], edge[1], int(edge[2])
        snap_edges.add((tuple(left), tuple(right), channel))

snap_listed = sorted([list(l), list(r), c] for (l, r, c) in snap_edges)
snap_hash = hashlib.sha256(json.dumps(snap_listed, sort_keys=True).encode()).hexdigest()

glob_listed, glob_hash = pinned_historical_edges()
glob_set = {(tuple(l), tuple(r), c) for (l, r, c) in glob_listed}

only_snap = snap_edges - glob_set
only_glob = glob_set - snap_edges

report = {
    "snapshot_edge_count": len(snap_edges),
    "snapshot_edges_sha256": snap_hash,
    "liveglob_edge_count": len(glob_listed),
    "liveglob_edges_sha256": glob_hash,
    "counts_equal": len(snap_edges) == len(glob_listed),
    "hashes_equal": snap_hash == glob_hash,
    "sets_identical": not only_snap and not only_glob,
    "edges_only_in_snapshot": len(only_snap),
    "edges_only_in_liveglob": len(only_glob),
    "sample_only_in_snapshot": sorted([list(l), list(r), c] for (l, r, c) in list(only_snap)[:5]),
    "sample_only_in_liveglob": sorted([list(l), list(r), c] for (l, r, c) in list(only_glob)[:5]),
}
print(json.dumps(report, indent=2))
