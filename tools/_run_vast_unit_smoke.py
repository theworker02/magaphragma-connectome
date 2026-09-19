#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from cloud.budget import BudgetConfig, BudgetLedger, BudgetRefused
from cloud.pricing import estimate_chunk_economics, fleet_size_for_target_hours, max_gpus_under_budget

m = estimate_chunk_economics(tiles_per_second=18.9, gpu_hourly_usd=0.1, remaining_chunks=1000)
assert m.chunks_per_hour > 0
print("cost_ok", round(m.effective_cost_per_chunk, 6), round(m.projected_remaining_cost_usd, 2))

n = fleet_size_for_target_hours(remaining_chunks=28598, tiles_per_second_per_gpu=18.9, target_hours=168, max_gpus=20)
assert 1 <= n <= 20
print("fleet_size", n)

assert max_gpus_under_budget(budget_usd=100, hours=24, dph_per_gpu=0.2) == int(100 // (0.2 * 24))

with tempfile.TemporaryDirectory() as td:
    led = BudgetLedger(Path(td) / "l.json", BudgetConfig(hard_limit_usd=10))
    try:
        led.check_projected_total(50, context="t")
        raise SystemExit("should have refused")
    except BudgetRefused:
        print("refuse_ok")
    led2 = BudgetLedger(Path(td) / "l2.json", BudgetConfig(hard_limit_usd=160))
    s = led2.authorize_provision(hourly_usd=0.5, hours=10, context="t", projected_remaining_cost_usd=50)
    assert abs(s["committed_usd"] - 5.0) < 1e-9
    led2.release_commitment(5.0, context="done")
    led3 = BudgetLedger(Path(td) / "l3.json", BudgetConfig(hard_limit_usd=100, stop_provision_frac=0.85))
    led3.record_spend(86.0, context="seed")
    try:
        led3.authorize_provision(hourly_usd=1.0, hours=1.0, context="t")
        raise SystemExit("should stop provision")
    except BudgetRefused:
        print("stop85_ok")

import os
from s7_chunk_claim import claim_backend, try_claim_dynamodb

os.environ["S7_CLAIM_BACKEND"] = "dynamodb"
try:
    try:
        claim_backend()
        raise SystemExit("ddb backend should raise")
    except RuntimeError:
        print("ddb_backend_ok")
    try:
        try_claim_dynamodb("x", "t")
        raise SystemExit("ddb claim should raise")
    except RuntimeError:
        print("ddb_claim_ok")
finally:
    os.environ["S7_CLAIM_BACKEND"] = "local"

print("ALL_PASS")
