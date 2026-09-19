"""Unit tests for Affinity Vast budget + cost model (no cloud calls)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from cloud.budget import BudgetConfig, BudgetLedger, BudgetRefused
from cloud.pricing import estimate_chunk_economics, fleet_size_for_target_hours, max_gpus_under_budget


class TestCostModel(unittest.TestCase):
    def test_effective_cost_per_chunk(self):
        m = estimate_chunk_economics(
            tiles_per_second=18.9,
            gpu_hourly_usd=0.10,
            remaining_chunks=1000,
        )
        self.assertGreater(m.chunks_per_hour, 0)
        self.assertAlmostEqual(m.seconds_per_chunk, 3468 / 18.9, places=3)
        self.assertLess(m.effective_cost_per_chunk, 1.0)
        self.assertGreater(m.projected_remaining_cost_usd, 0)

    def test_fleet_size(self):
        n = fleet_size_for_target_hours(
            remaining_chunks=28598,
            tiles_per_second_per_gpu=18.9,
            target_hours=168,
            max_gpus=20,
        )
        self.assertGreaterEqual(n, 1)
        self.assertLessEqual(n, 20)

    def test_max_gpus_under_budget(self):
        n = max_gpus_under_budget(budget_usd=100, hours=24, dph_per_gpu=0.2)
        self.assertEqual(n, int(100 // (0.2 * 24)))


class TestBudgetLedger(unittest.TestCase):
    def test_refuse_over_hard_limit(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ledger.json"
            led = BudgetLedger(path, BudgetConfig(hard_limit_usd=10, target_usd=8, benchmark_usd=2))
            with self.assertRaises(BudgetRefused):
                led.check_projected_total(50.0, context="test")

    def test_stop_provision_at_85pct(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ledger.json"
            led = BudgetLedger(path, BudgetConfig(hard_limit_usd=100, stop_provision_frac=0.85))
            led.record_spend(86.0, context="seed")
            with self.assertRaises(BudgetRefused):
                led.authorize_provision(hourly_usd=1.0, hours=1.0, context="test")

    def test_authorize_then_release(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ledger.json"
            led = BudgetLedger(path, BudgetConfig(hard_limit_usd=160))
            snap = led.authorize_provision(hourly_usd=0.5, hours=10, context="t", projected_remaining_cost_usd=50)
            self.assertAlmostEqual(snap["committed_usd"], 5.0)
            snap2 = led.release_commitment(5.0, context="done")
            self.assertAlmostEqual(snap2["committed_usd"], 0.0)


class TestDynamodbDecommissioned(unittest.TestCase):
    def test_claim_backend_rejects_dynamodb(self):
        import os

        from s7_chunk_claim import claim_backend, try_claim_dynamodb

        os.environ["S7_CLAIM_BACKEND"] = "dynamodb"
        try:
            with self.assertRaises(RuntimeError):
                claim_backend()
            with self.assertRaises(RuntimeError):
                try_claim_dynamodb("x", "table")
        finally:
            os.environ["S7_CLAIM_BACKEND"] = "local"


if __name__ == "__main__":
    unittest.main()
