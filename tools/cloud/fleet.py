"""Fleet planning and economic auto-scale helpers (no silent budget bypass)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .base import Offer
from .budget import BudgetLedger, BudgetRefused
from .pricing import (
    CostModel,
    estimate_chunk_economics,
    fleet_size_for_target_hours,
    max_gpus_under_budget,
)
from .vast_provider import VastProvider


@dataclass
class FleetPlan:
    qualified_gpu: str
    n_gpus: int
    offer_ids: list[int]
    dph_per_gpu: float
    fleet_hourly_usd: float
    tiles_per_sec_per_gpu: float
    aggregate_tiles_per_sec: float
    chunks_per_hour: float
    remaining_chunks: int
    estimated_completion_hours: float
    estimated_compute_cost_usd: float
    estimated_ancillary_cost_usd: float
    estimated_total_cost_usd: float
    budget_remaining_usd: float
    interruptible: bool
    cost_per_chunk: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_fleet_plan(
    *,
    provider: VastProvider,
    ledger: BudgetLedger,
    remaining_chunks: int,
    tiles_per_second: float,
    budget_usd: float | None = None,
    target_hours: float = 168.0,
    max_gpus: int = 20,
    max_price_per_hour: float | None = None,
    min_reliability: float = 0.98,
    storage_hourly_per_gpu: float = 0.0,
    bandwidth_per_chunk_usd: float = 0.0,
    prefer_interruptible: bool = False,
) -> FleetPlan:
    """
    Plan only — does NOT launch.

    Uses live offers + measured (or baseline) TPS. Refuses if projected cost
    exceeds hard budget.
    """
    snap = ledger.snapshot()
    hard = ledger.config.hard_limit_usd
    budget = float(budget_usd if budget_usd is not None else min(ledger.config.target_usd, hard))
    budget = min(budget, snap["remaining_usd"], hard)

    offers = provider.search_offers_both_markets(
        min_reliability=min_reliability,
        max_dph=max_price_per_hour,
        limit=80,
    )
    if not offers:
        raise RuntimeError("No Vast offers matched Affinity filters")

    ranked = provider.rank_offers(
        offers,
        tiles_per_second_by_gpu={},
        default_tiles_per_second=tiles_per_second,
        remaining_chunks=remaining_chunks,
        bandwidth_per_chunk_usd=bandwidth_per_chunk_usd,
    )
    # Prefer non-interruptible unless asked; still keep cheapest projected cost
    if not prefer_interruptible:
        od = [o for o in ranked if not o.interruptible]
        if od:
            ranked = od + [o for o in ranked if o.interruptible]

    best = ranked[0]
    model = estimate_chunk_economics(
        tiles_per_second=tiles_per_second,
        gpu_hourly_usd=best.dph_total,
        remaining_chunks=remaining_chunks,
        storage_hourly_usd=storage_hourly_per_gpu,
        bandwidth_per_chunk_usd=bandwidth_per_chunk_usd,
    )
    ledger.check_projected_total(model.projected_remaining_cost_usd, context="fleet_plan")

    need = fleet_size_for_target_hours(
        remaining_chunks=remaining_chunks,
        tiles_per_second_per_gpu=tiles_per_second,
        target_hours=target_hours,
        max_gpus=max_gpus,
    )
    affordable = max_gpus_under_budget(
        budget_usd=budget,
        hours=min(target_hours, model.projected_remaining_gpu_hours),
        dph_per_gpu=best.dph_total + storage_hourly_per_gpu,
    )
    n = max(0, min(need, affordable, max_gpus))
    if n < 1:
        raise BudgetRefused(
            f"Cannot afford even 1 GPU under budget=${budget:.2f} "
            f"at ${best.dph_total:.4f}/h for target_hours={target_hours}"
        )

    # Collect up to n offers of same GPU family / similar price
    selected: list[Offer] = []
    for o in ranked:
        if o.gpu_name != best.gpu_name and abs(o.dph_total - best.dph_total) > best.dph_total * 0.25:
            continue
        if max_price_per_hour is not None and o.dph_total > max_price_per_hour:
            continue
        selected.append(o)
        if len(selected) >= n:
            break
    if not selected:
        selected = [best]
        n = 1

    fleet_hourly = sum(o.dph_total for o in selected) + storage_hourly_per_gpu * len(selected)
    agg_tps = tiles_per_second * len(selected)
    chunks_per_hour = (agg_tps / model.tiles_per_chunk) * 3600.0 if model.tiles_per_chunk else 0.0
    completion_h = remaining_chunks / chunks_per_hour if chunks_per_hour > 0 else float("inf")
    compute_cost = fleet_hourly * completion_h
    ancillary = bandwidth_per_chunk_usd * remaining_chunks
    total = compute_cost + ancillary

    ledger.check_projected_total(total, context="fleet_plan_total")
    if total > hard:
        raise BudgetRefused(f"REFUSE TO SCALE: estimated total ${total:.2f} > hard ${hard:.2f}")

    notes = [
        f"best_offer_id={best.offer_id}",
        f"need_for_target={need}",
        f"affordable_under_budget={affordable}",
        f"ranking_metric=projected_remaining_cost",
    ]
    if best.interruptible:
        notes.append("interruptible=true; requires claim expiry + restart safety")

    return FleetPlan(
        qualified_gpu=best.gpu_name,
        n_gpus=len(selected),
        offer_ids=[o.offer_id for o in selected],
        dph_per_gpu=best.dph_total,
        fleet_hourly_usd=fleet_hourly,
        tiles_per_sec_per_gpu=tiles_per_second,
        aggregate_tiles_per_sec=agg_tps,
        chunks_per_hour=chunks_per_hour,
        remaining_chunks=remaining_chunks,
        estimated_completion_hours=completion_h,
        estimated_compute_cost_usd=compute_cost,
        estimated_ancillary_cost_usd=ancillary,
        estimated_total_cost_usd=total,
        budget_remaining_usd=snap["remaining_usd"],
        interruptible=best.interruptible,
        cost_per_chunk=model.effective_cost_per_chunk,
        notes=notes,
    )


def cost_model_from_benchmark(
    *,
    tiles_per_second: float,
    gpu_hourly_usd: float,
    remaining_chunks: int,
    **kwargs: Any,
) -> CostModel:
    return estimate_chunk_economics(
        tiles_per_second=tiles_per_second,
        gpu_hourly_usd=gpu_hourly_usd,
        remaining_chunks=remaining_chunks,
        **kwargs,
    )
