#!/usr/bin/env python3
"""HyperDrain fleet watchdog — status, throughput, measured ETA."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from hyperdrain import config, queue  # noqa: E402


def gather_throughput(limit: int = 50) -> dict:
    chunks_dir = config.OUT / "chunks"
    if not chunks_dir.exists():
        return {"n_samples": 0, "mean_wall_s": None, "mean_tiles_per_sec": None}
    samples = []
    for tel in sorted(chunks_dir.glob("*/throughput_telemetry.json"), key=lambda p: p.stat().st_mtime)[-limit:]:
        try:
            t = json.loads(tel.read_text(encoding="utf-8"))
            wall = (t.get("phase_seconds") or {}).get("total_wall_seconds") or t.get("infer_seconds")
            samples.append(
                {
                    "tiles_per_sec": t.get("tiles_per_second"),
                    "wall_s": wall,
                    "infer_s": t.get("infer_seconds"),
                }
            )
        except Exception:
            continue
    if not samples:
        return {"n_samples": 0, "mean_wall_s": None, "mean_tiles_per_sec": None}
    walls = [s["wall_s"] for s in samples if s["wall_s"]]
    tps = [s["tiles_per_sec"] for s in samples if s["tiles_per_sec"]]
    return {
        "n_samples": len(samples),
        "mean_wall_s": sum(walls) / len(walls) if walls else None,
        "mean_tiles_per_sec": sum(tps) / len(tps) if tps else None,
        "preserved_baselines": {
            "eager": config.BASELINE_EAGER_TILES_PER_SEC,
            "compile": config.BASELINE_COMPILE_TILES_PER_SEC,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="HyperDrain watchdog")
    ap.add_argument("--watch", action="store_true", help="Poll forever")
    ap.add_argument("--interval", type=float, default=30.0)
    args = ap.parse_args()
    config.ensure_out()
    if not config.QUEUE_STATE.exists():
        queue.bootstrap_from_s7()

    def once() -> None:
        summary = queue.status_summary()
        thr = gather_throughput()
        remaining = summary["n_total"] - summary["n_done"]
        eta_h = None
        if thr.get("mean_wall_s") and remaining > 0:
            eta_h = remaining * float(thr["mean_wall_s"]) / 3600.0
        report = {
            "queue": summary,
            "throughput": thr,
            "eta_hours_one_worker_measured": eta_h,
            "note": "ETA from measured HyperDrain walls only; not fabricated.",
        }
        print(json.dumps(report, indent=2), flush=True)

    if not args.watch:
        once()
        return 0
    while True:
        once()
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
