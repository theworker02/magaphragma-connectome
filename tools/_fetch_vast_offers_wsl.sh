#!/usr/bin/env bash
# Fetch Vast offers using env VAST_API_KEY (never echoed).
set -euo pipefail
ROOT="/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome"
OUT="$ROOT/experiments/phase6e/VAST-AFFINITY-BENCH-001"
mkdir -p "$OUT"
if [[ -z "${VAST_API_KEY:-}" ]]; then
  echo '{"ok":false,"error":"VAST_API_KEY missing in this shell"}' >&2
  exit 4
fi
TMP=$(mktemp)
curl -sS -H "Authorization: Bearer ${VAST_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"verified":{"eq":true},"rentable":{"eq":true},"num_gpus":{"eq":1},"gpu_ram":{"gte":6144},"reliability":{"gte":0.98},"disk_space":{"gte":32},"cpu_ram":{"gte":8192},"direct_port_count":{"gte":1},"type":"on-demand","limit":60,"order":[["dph_total","asc"]]}' \
  "https://console.vast.ai/api/v0/bundles/" > "$TMP"
# Parse + rank with python (stdlib only) writing receipt
python3 - "$TMP" "$OUT" <<'PY'
import json, sys
from pathlib import Path
raw_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
payload = json.loads(raw_path.read_text())
offers = payload.get("offers") or payload.get("bundles") or []
TILES = 3468
TPS = 18.9
# remaining chunks from progress if readable
rem = 28598
prog = Path("/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome/experiments/phase6e/AFFINITY-FULLVOL-S7-001/PROGRESS.json")
if prog.exists():
    rem = int(json.loads(prog.read_text()).get("n_not_started", rem))
rows = []
for row in offers:
    if not isinstance(row, dict):
        continue
    gpu_ram = float(row.get("gpu_ram") or 0)
    vram_gb = gpu_ram / 1024.0 if gpu_ram > 64 else gpu_ram
    dph = float(row.get("dph_total") or 0)
    cph = (TPS / TILES) * 3600.0
    cpc = dph / cph if cph else None
    proj = (cpc or 0) * rem
    rows.append({
        "offer_id": row.get("id"),
        "gpu_name": row.get("gpu_name"),
        "vram_gb": round(vram_gb, 2),
        "dph_total": dph,
        "reliability": row.get("reliability"),
        "interruptible": False,
        "verified": row.get("verified"),
        "geolocation": row.get("geolocation"),
        "assumed_tiles_per_second": TPS,
        "effective_cost_per_chunk": cpc,
        "projected_remaining_cost_usd": proj,
    })
rows.sort(key=lambda r: (r["projected_remaining_cost_usd"], r["dph_total"]))
doc = {
    "remaining_chunks": rem,
    "assumed_tiles_per_second": TPS,
    "n_offers": len(rows),
    "offer_type": "on-demand",
    "note": "TPS assumed = local RX 7800 XT baseline 18.9 until Vast GPU measured",
    "offers": rows[:50],
}
(out_dir / "VAST_GPU_OFFERS.json").write_text(json.dumps(doc, indent=2) + "\n")
if rows:
    top = rows[0]
    cpc = top["effective_cost_per_chunk"]
    proj = top["projected_remaining_cost_usd"]
    (out_dir / "VAST_COST_MODEL.json").write_text(json.dumps({
        "id": "VAST_COST_MODEL",
        "status": "PRELIMINARY_UNMEASURED_TPS",
        "warning": "tiles/sec not yet measured on this Vast GPU",
        "top_offer": top,
        "model": {
            "validated_tiles_per_second": TPS,
            "gpu_hourly_usd": top["dph_total"],
            "remaining_chunks": rem,
            "effective_cost_per_chunk": cpc,
            "projected_remaining_cost_usd": proj,
            "projected_remaining_gpu_hours": rem * TILES / TPS / 3600.0,
            "cost_per_1000_chunks": cpc * 1000,
            "seconds_per_chunk": TILES / TPS,
            "chunks_per_hour": (TPS / TILES) * 3600.0,
        },
        "budgets": {"hard": 160, "target": 100, "benchmark": 5},
        "within_hard_budget": proj <= 160,
        "within_target_budget": proj <= 100,
    }, indent=2) + "\n")
    (out_dir / "VAST_SELECTED_GPU.json").write_text(json.dumps({
        "status": "CANDIDATE_UNQUALIFIED",
        "reason": "Cheapest live on-demand offer under assumed TPS; must pass $5 Vast qualification before fleet",
        "offer_id": top["offer_id"],
        "gpu_name": top["gpu_name"],
        "dph_total": top["dph_total"],
        "projected_remaining_cost_usd_assumed_tps": proj,
    }, indent=2) + "\n")
print(json.dumps({"n_offers": len(rows), "top5": rows[:5]}, indent=2))
PY
rm -f "$TMP"
# also try interruptible
TMP2=$(mktemp)
curl -sS -H "Authorization: Bearer ${VAST_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"verified":{"eq":true},"rentable":{"eq":true},"num_gpus":{"eq":1},"gpu_ram":{"gte":6144},"reliability":{"gte":0.98},"type":"bid","limit":30,"order":[["dph_total","asc"]]}' \
  "https://console.vast.ai/api/v0/bundles/" > "$TMP2" || true
python3 - "$TMP2" "$OUT" <<'PY'
import json, sys
from pathlib import Path
raw, out = Path(sys.argv[1]), Path(sys.argv[2])
try:
    payload = json.loads(raw.read_text())
except Exception:
    payload = {}
offers = payload.get("offers") or []
(out / "VAST_GPU_OFFERS_INTERRUPTIBLE_RAW_COUNT.json").write_text(json.dumps({"n": len(offers)}, indent=2)+ "\n")
print("interruptible_offers", len(offers))
PY
rm -f "$TMP2"
