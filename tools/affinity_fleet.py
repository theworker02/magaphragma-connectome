#!/usr/bin/env python3
"""Unified Affinity fleet status (local + Vast workers via coordinator)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

OUT = Path(os.environ.get("S7_OUT", str(REPO / "experiments/phase6e/AFFINITY-FULLVOL-S7-001")))
TILES_PER_CHUNK = 3468
COST_PATH = REPO / "experiments/phase6e/VAST-AFFINITY-BENCH-001" / "VAST_FLEET_COST.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def local_counts() -> dict:
    state_path = OUT / "queue_state.json"
    if not state_path.exists():
        return {}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for st in state.get("status_by_id", {}).values():
        counts[st] = counts.get(st, 0) + 1
    return counts


def fetch_coordinator(url: str) -> dict | None:
    try:
        with urlopen(url.rstrip("/") + "/status", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": str(e)}


def load_cost() -> dict:
    if COST_PATH.exists():
        return json.loads(COST_PATH.read_text(encoding="utf-8"))
    return {}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Affinity hybrid fleet status")
    ap.add_argument("command", nargs="?", default="status", choices=["status"], help="status (default)")
    ap.add_argument("--claim-http", default=os.environ.get("S7_CLAIM_HTTP", "http://127.0.0.1:8787"))
    return ap


def main() -> int:
    args = build_parser().parse_args()
    # command currently only status
    _ = args.command

    counts = local_counts()
    n_total = sum(counts.values()) or 28798
    n_done = counts.get("COMPLETE", 0) + counts.get("AFFINITY_DONE_SEG_PENDING", 0)
    n_run = counts.get("RUNNING", 0)
    n_left = counts.get("NOT_STARTED", 0)

    coord = fetch_coordinator(args.claim_http) if args.claim_http else None
    workers = {}
    telem = {}
    if coord and coord.get("ok"):
        workers = coord.get("workers") or {}
        telem = coord.get("telemetry") or {}
        if coord.get("counts"):
            counts = coord["counts"]
            n_total = sum(counts.values()) or n_total
            n_done = counts.get("COMPLETE", 0) + counts.get("AFFINITY_DONE_SEG_PENDING", 0)
            n_run = counts.get("RUNNING", 0)
            n_left = counts.get("NOT_STARTED", 0)

    # Also merge local fleet files if coordinator down
    wp = OUT / "fleet_workers.json"
    tp = OUT / "fleet_telemetry.json"
    if wp.exists() and not workers:
        workers = json.loads(wp.read_text(encoding="utf-8"))
    if tp.exists() and not telem:
        telem = json.loads(tp.read_text(encoding="utf-8"))

    agg_tps = 0.0
    lines = []
    lines.append("AFFINITY PRODUCTION")
    lines.append("")
    lines.append(f"Total chunks:     {n_total}")
    lines.append(f"Completed:        {n_done}")
    lines.append(f"Running:          {n_run}")
    lines.append(f"Remaining:        {n_left}")
    lines.append("")
    lines.append("Workers:")
    lines.append("")

    for wid, meta in sorted(workers.items()):
        t = telem.get(wid) or {}
        tps = float(t.get("tiles_per_second") or 0.0)
        if tps:
            agg_tps += tps
        cph = (tps / TILES_PER_CHUNK) * 3600.0 if tps else 0.0
        cloud_cost = float(meta.get("dph_usd") or 0.0)
        lines.append(wid)
        lines.append(f"  GPU: {meta.get('gpu', '?')}")
        lines.append(f"  runtime: {meta.get('runtime', '?')}")
        lines.append(f"  status: {t.get('status', meta.get('status', 'REGISTERED'))}")
        lines.append(f"  tiles/sec: {tps:.2f}" if tps else "  tiles/sec: (no telemetry yet)")
        lines.append(f"  chunks/hour: {cph:.2f}" if tps else "  chunks/hour: n/a")
        lines.append(f"  cloud $/hour: {cloud_cost:.4f}" if cloud_cost else "  cloud $/hour: 0 (local)")
        lines.append("")

    if not workers:
        lines.append("  (none registered — start claim coordinator + workers)")
        lines.append("")

    agg_cph = (agg_tps / TILES_PER_CHUNK) * 3600.0 if agg_tps else 0.0
    eta_h = (n_left / agg_cph) if agg_cph > 0 else None
    lines.append("Aggregate:")
    lines.append(f"  tiles/sec: {agg_tps:.2f}")
    lines.append(f"  chunks/hour: {agg_cph:.2f}")
    if eta_h is not None:
        lines.append(f"Estimated completion: {eta_h:.1f} hours ({eta_h/24:.2f} days)")
    else:
        lines.append("Estimated completion: n/a (no active TPS telemetry)")

    cost = load_cost()
    lines.append("")
    lines.append(f"Cloud spend (recorded): ${float(cost.get('accrued_usd', 0.0)):.4f}")
    if agg_cph > 0 and cost.get("fleet_hourly_usd"):
        proj = float(cost["fleet_hourly_usd"]) * (eta_h or 0)
        lines.append(f"Projected remaining cloud spend: ${proj:.2f}")
    lines.append(f"Updated: {_now()}")

    text = "\n".join(lines) + "\n"
    print(text)
    (OUT / "FLEET_STATUS.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
