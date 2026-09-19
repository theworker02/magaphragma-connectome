#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter
from urllib.request import urlopen

OUT = Path("/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome/experiments/phase6e/AFFINITY-FULLVOL-S7-001")
chunks = OUT / "chunks"
rows = []
for d in sorted(chunks.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:25]:
    rpath = d / "receipt.json"
    if not rpath.exists():
        continue
    r = json.loads(rpath.read_text())
    tel = r.get("throughput_telemetry") or {}
    rows.append({
        "chunk": d.name,
        "worker_id": r.get("worker_id"),
        "tps": tel.get("tiles_per_second"),
        "batch": tel.get("tile_batch_size"),
        "durable": r.get("durable_commit"),
        "mtime": d.stat().st_mtime,
    })
print("RECENT_CHUNKS")
print(json.dumps(rows, indent=2))
print("WORKER_COUNTS", Counter(r.get("worker_id") or "?" for r in rows))
try:
    st = json.loads(urlopen("http://127.0.0.1:8787/status", timeout=5).read().decode())
    print("COORD", json.dumps({k: st.get(k) for k in ("counts", "workers", "telemetry", "n_total")}, indent=2))
except Exception as e:
    print("COORD_ERR", e)

claims = list((OUT / "claims").glob("*.claim.json")) if (OUT / "claims").exists() else []
print("N_ACTIVE_CLAIMS", len(claims))
for c in claims[:10]:
    print(c.name, c.read_text()[:200].replace("\n", " "))
