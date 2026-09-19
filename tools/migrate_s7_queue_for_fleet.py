#!/usr/bin/env python3
"""Migrate Affinity queue for hybrid fleet: reclaim stale RUNNING, preserve done work.

Conservative: only returns RUNNING -> NOT_STARTED when affinities are absent
and claim is missing/stale. Never deletes completed artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from s7_chunk_claim import OUT, load_state, reclaim_stale_running, save_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age-s", type=float, default=1800.0, help="Reclaim RUNNING older than this (default 30m)")
    ap.add_argument("--force-all-running", action="store_true", help="Reclaim ALL RUNNING without affinities regardless of age")
    args = ap.parse_args()

    state = load_state()
    before = {cid: st for cid, st in state.get("status_by_id", {}).items() if st == "RUNNING"}
    if args.force_all_running:
        reclaim_stale_running(state, max_age_s=0.0)
    else:
        reclaim_stale_running(state, max_age_s=args.max_age_s)
    save_state(state)
    after = state.get("status_by_id", {})
    reclaimed = [cid for cid in before if after.get(cid) == "NOT_STARTED"]
    kept = [cid for cid in before if after.get(cid) == "RUNNING"]
    counts: dict[str, int] = {}
    for st in after.values():
        counts[st] = counts.get(st, 0) + 1
    report = {
        "out": str(OUT),
        "running_before": sorted(before),
        "reclaimed": sorted(reclaimed),
        "still_running": sorted(kept),
        "counts": counts,
        "note": "Completed / AFFINITY_DONE_SEG_PENDING unchanged. Artifacts untouched.",
    }
    print(json.dumps(report, indent=2))
    (OUT / "FLEET_QUEUE_MIGRATE.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
