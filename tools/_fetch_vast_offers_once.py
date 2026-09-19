#!/usr/bin/env python3
"""Fetch live Vast offers; never print API key."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

key = os.environ.get("VAST_API_KEY", "").strip()
if not key:
    # common local file used by vastai CLI — read without echoing
    for candidate in (
        Path.home() / ".vast_api_key",
        Path.home() / ".config" / "vastai" / "vast_api_key",
    ):
        if candidate.is_file():
            key = candidate.read_text(encoding="utf-8").strip()
            os.environ["VAST_API_KEY"] = key
            break
if not key:
    print(json.dumps({"ok": False, "error": "VAST_API_KEY not set"}))
    raise SystemExit(4)

from cloud.benchmark import remaining_chunks
from cloud.pricing import BASELINE_TILES_PER_SEC, estimate_chunk_economics
from cloud.vast_provider import VastProvider

BENCH = REPO / "experiments/phase6e/VAST-AFFINITY-BENCH-001"
BENCH.mkdir(parents=True, exist_ok=True)

p = VastProvider(api_key=key)
print("searching offers...", flush=True)
offers = p.search_offers_both_markets(limit=80)
print(f"got {len(offers)} offers", flush=True)
rem = remaining_chunks()
ranked = p.rank_offers(offers, default_tiles_per_second=BASELINE_TILES_PER_SEC, remaining_chunks=rem)
rows = []
for o in ranked[:50]:
    rows.append(
        {
            "offer_id": o.offer_id,
            "gpu_name": o.gpu_name,
            "vram_gb": round(o.vram_gb, 2),
            "dph_total": o.dph_total,
            "reliability": o.reliability,
            "interruptible": o.interruptible,
            "verified": o.verified,
            "geolocation": o.geolocation,
            "assumed_tiles_per_second": o.raw.get("_assumed_tiles_per_second"),
            "effective_cost_per_chunk": o.raw.get("_effective_cost_per_chunk"),
            "projected_remaining_cost_usd": o.raw.get("_projected_remaining_cost_usd"),
        }
    )
payload = {
    "remaining_chunks": rem,
    "assumed_tiles_per_second": BASELINE_TILES_PER_SEC,
    "n_offers": len(rows),
    "note": "TPS assumed = RX 7800 XT baseline until a Vast GPU is measured; rank by projected cost under that assumption",
    "offers": rows,
}
out = BENCH / "VAST_GPU_OFFERS.json"
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

# Also write preliminary cost model from cheapest offer under assumed TPS
if rows:
    top = rows[0]
    model = estimate_chunk_economics(
        tiles_per_second=BASELINE_TILES_PER_SEC,
        gpu_hourly_usd=float(top["dph_total"]),
        remaining_chunks=rem,
    )
    (BENCH / "VAST_COST_MODEL.json").write_text(
        json.dumps(
            {
                "id": "VAST_COST_MODEL",
                "status": "PRELIMINARY_UNMEASURED_TPS",
                "warning": "tiles/sec not yet measured on this Vast GPU — using local baseline 18.9",
                "top_offer": top,
                "model": model.to_dict(),
                "budgets": {"hard": 160, "target": 100, "benchmark": 5},
                "within_hard_budget": model.projected_remaining_cost_usd <= 160,
                "within_target_budget": model.projected_remaining_cost_usd <= 100,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (BENCH / "VAST_SELECTED_GPU.json").write_text(
        json.dumps(
            {
                "status": "CANDIDATE_UNQUALIFIED",
                "reason": "Cheapest live offer under assumed TPS; must pass $5 Vast qualification before fleet launch",
                "offer_id": top["offer_id"],
                "gpu_name": top["gpu_name"],
                "dph_total": top["dph_total"],
                "projected_remaining_cost_usd_assumed_tps": top["projected_remaining_cost_usd"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

print(json.dumps({"wrote": str(out), "n_offers": len(rows), "top5": rows[:5]}, indent=2))
