#!/usr/bin/env python3
"""Print compact Vast fleet inventory (no secrets)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from cloud.vast_provider import VastProvider, load_repo_dotenv  # noqa: E402

load_repo_dotenv(REPO)
p = VastProvider()
fs = p.fleet_status()
rows = []
gpus = 0
for i in fs.get("instances", []):
    raw = i.get("raw") or {}
    ng = int(raw.get("num_gpus") or 1)
    gpus += ng
    pip = raw.get("public_ipaddr")
    port = None
    ports = raw.get("ports") or {}
    if "22/tcp" in ports and ports["22/tcp"]:
        port = ports["22/tcp"][0].get("HostPort")
    if not port:
        port = i.get("ssh_port")
    rows.append(
        {
            "id": i["instance_id"],
            "status": i.get("actual_status"),
            "gpu": i.get("gpu_name"),
            "n": ng,
            "dph": round(float(i.get("dph_total") or 0), 4),
            "direct": f"{pip}:{port}" if pip and port else None,
            "proxy": f"{i.get('ssh_host')}:{i.get('ssh_port')}",
            "label": i.get("label"),
        }
    )
out = {
    "n_instances": fs.get("n_instances"),
    "n_running": fs.get("n_running"),
    "n_loading": fs.get("n_loading"),
    "fleet_hourly_usd": round(float(fs.get("fleet_hourly_usd") or 0), 4),
    "total_gpus": gpus,
    "instances": rows,
}
print(json.dumps(out, indent=2))
