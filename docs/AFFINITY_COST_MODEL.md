# Affinity cost model

Primary optimization metric:

**dollars per completed production chunk**

Not GPU prestige, not VRAM capacity, not raw tiles/sec alone.

## Definitions

Production tiles/chunk (FAST geometry): **3468**  
Crop ZYX `(20,64,64)` · Stride ZYX `(10,64,64)`

```text
seconds_per_chunk     = 3468 / validated_tiles_per_second
chunks_per_hour       = 3600 / seconds_per_chunk
compute_cost_per_chunk = (gpu_hourly_usd + storage_hourly_usd) / chunks_per_hour
effective_cost_per_chunk = compute_cost_per_chunk + bandwidth_per_chunk_usd
projected_remaining_cost = remaining_chunks * effective_cost_per_chunk
```

Rank marketplace offers primarily by **projected_remaining_cost**.

Also report: `$/1000 chunks`, tiles/sec, seconds/chunk, chunks/hour, GPU $/hour, storage, bandwidth, wall ETA, reliability, interruption risk.

## Baselines

| Source | Notes |
|---|---|
| RX 7800 XT local | ~18.9 tiles/s packed-eager reference; **$0** cloud rental |
| Peak VRAM | ~3.3 GiB measured → search ≥6 GB (prefer ≥8 GB) |
| Remaining workload | from `AFFINITY-FULLVOL-S7-001/PROGRESS.json` `n_not_started` |

## Budgets

| Limit | USD |
|---|---|
| Hard | 160 |
| Target | 100 |
| Benchmark campaign | 5 |

Ledger: `experiments/phase6e/VAST-AFFINITY-BENCH-001/budget_ledger.json`

Phases: OK → WARNING (70%) → STOP_PROVISIONING (85%) → DRAIN_AND_DESTROY (95%) → HARD_STOP (100%).

## Implementation

- `tools/cloud/pricing.py` — CostModel
- `tools/cloud/budget.py` — BudgetLedger / BudgetRefused
- `tools/cloud/fleet.py` — plan from live offers
- `tools/affinity_vast.py` — CLI

Do not assume 10–12 physical GPUs. Fleet size is derived from measured TPS, offer price, budget, and target hours.
