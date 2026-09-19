#!/usr/bin/env python3
"""Print compact Vast SSH endpoints (no secrets)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
out = subprocess.check_output(
    [sys.executable, str(REPO / "tools" / "affinity_vast.py"), "status"],
    text=True,
    errors="replace",
    cwd=str(REPO),
)
d = json.loads(out)
for i in d["fleet"]["instances"]:
    r = i.get("raw") or {}
    print(
        "|".join(
            [
                str(i["instance_id"]),
                str(i.get("gpu_name")),
                f"{i['ssh_host']}:{i['ssh_port']}",
                f"util={r.get('gpu_util')}",
                f"label={i.get('label')}",
                f"ip={r.get('public_ipaddr')}",
            ]
        )
    )
print(f"n={d['fleet']['n_instances']} hourly={d['fleet']['fleet_hourly_usd']}")
