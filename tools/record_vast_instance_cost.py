#!/usr/bin/env python3
"""Record Vast instance hourly price into fleet cost ledger (do not guess)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "experiments/phase6e/VAST-AFFINITY-BENCH-001" / "VAST_FLEET_COST.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--dph", type=float, required=True, help="Actual $/hour from Vast UI/API")
    ap.add_argument("--worker-id", default="vast-rtx3060ti-01")
    args = ap.parse_args()
    PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(PATH.read_text(encoding="utf-8")) if PATH.exists() else {"instances": {}, "accrued_usd": 0.0}
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    data["instances"][args.instance_id] = {
        "worker_id": args.worker_id,
        "gpu": args.gpu,
        "dph_usd": args.dph,
        "started_at": data.get("instances", {}).get(args.instance_id, {}).get("started_at") or now,
        "updated_at": now,
    }
    data["fleet_hourly_usd"] = sum(float(i["dph_usd"]) for i in data["instances"].values())
    data["updated_at"] = now
    PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
