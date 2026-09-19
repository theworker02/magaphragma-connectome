"""Self-contained triple determinism check for the -008 snapshot-only sampler.
Runs select_only() three times in-process (independent computations) and
compares canonical manifests exactly. Also confirms the on-disk -008 queues
match the canonical selection and all event logs are empty. Writes result to
the OS temp dir to avoid OneDrive write races in the repo tree."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_g3_008_axis_neutral as B

REPO = Path(__file__).resolve().parents[1]
Q_ROOT = REPO / "experiments/phase6e/g3-interface-queues-008"
WS_ROOT = REPO / "experiments/phase6e/g3-external-review-packages-008"


def run():
    m1 = B.canonical_manifest(B.select_only())
    m2 = B.canonical_manifest(B.select_only())
    m3 = B.canonical_manifest(B.select_only())
    result = {
        "run1_eq_run2": m1 == m2,
        "run1_eq_run3": m1 == m3,
        "centers": len(m1["selection"]),
        "edges": len(m1["selection"]) * 3,
        "pinned_exclusion_sha256": m1["pinned_exclusion_sha256"],
        "pinned_exclusion_count": m1["pinned_exclusion_count"],
    }
    import collections
    result["strata"] = dict(collections.Counter(x["stratum"] for x in m1["selection"]))

    # Compare on-disk queues (12) to canonical selection edges.
    canon_edges = set()
    for it in m1["selection"]:
        z, y, x = it["center_zyx"]
        canon_edges.add((tuple([z, y, x]), tuple([z + 1, y, x]), 0))
        canon_edges.add((tuple([z, y, x]), tuple([z, y + 1, x]), 1))
        canon_edges.add((tuple([z, y, x]), tuple([z, y, x + 1]), 2))
    disk_edges = set()
    logs_empty = True
    queue_count = 0
    for qf in sorted(Q_ROOT.glob("MV-G3-AXNEU4-*.json")):
        queue_count += 1
        q = json.loads(qf.read_text())
        for qq in q["questions"]:
            disk_edges.add((tuple(qq["pair_left_zyx"]), tuple(qq["pair_right_zyx"]), int(qq["channel_zyx"])))
    for lg in WS_ROOT.glob("MV-G3-AXNEU4-*/workspace.events.jsonl"):
        if lg.read_text().strip() != "":
            logs_empty = False
    result["queue_count"] = queue_count
    result["disk_queues_match_canonical"] = (disk_edges == canon_edges)
    result["disk_edge_count"] = len(disk_edges)
    result["all_event_logs_empty"] = logs_empty

    out = Path(tempfile.gettempdir()) / "verify008.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(str(out))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
