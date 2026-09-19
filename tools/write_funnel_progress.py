#!/usr/bin/env python3
"""Write a lightweight funnel progress snapshot (local queue only).

AWS Affinity path is DECOMMISSIONED — this tool no longer queries ASG/S3/quotas.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/phase6e/AFFINITY-FULLVOL-S7-001"


def main() -> int:
    state = json.loads((OUT / "queue_state.json").read_text(encoding="utf-8"))
    counts = dict(Counter(state["status_by_id"].values()))
    snap = {
        "id": "AFFINITY_FUNNEL_PROGRESS",
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cloud_provider": "vast.ai",
        "aws_affinity": "DECOMMISSIONED",
        "local_status_counts": counts,
        "local_labels_npy": sum(1 for _ in (OUT / "chunks").glob("*/labels.npy")),
        "local_affinities_npy": sum(1 for _ in (OUT / "chunks").glob("*/affinities_core_czyx.npy")),
    }
    path = OUT / "FUNNEL_PROGRESS.json"
    path.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(snap, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
