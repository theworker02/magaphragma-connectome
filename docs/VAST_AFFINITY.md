# Affinity on Vast.ai

**Production cloud provider:** [Vast.ai](https://vast.ai/)  
**Local:** AMD ROCm (RX 7800 XT) remains a first-class free worker  
**AWS:** DECOMMISSIONED for Affinity production — see [AWS_DECOMMISSION.md](AWS_DECOMMISSION.md)

## Hard budgets

| Env | Default | Meaning |
|---|---|---|
| `AFFINITY_CLOUD_BUDGET_USD` | 160 | Absolute maximum cloud compute spend |
| `AFFINITY_TARGET_BUDGET_USD` | 100 | Preferred target |
| `AFFINITY_BENCHMARK_BUDGET_USD` | 5 | Max for qualification campaign |

The fleet controller estimates projected remaining cost **before** scaling. If projected cost exceeds the hard budget, it **refuses to scale**. There is no silent override.

## Setup

```bash
# Never commit the key
export VAST_API_KEY=...          # from https://cloud.vast.ai/account/
export AFFINITY_CLOUD_BUDGET_USD=160
export AFFINITY_TARGET_BUDGET_USD=100
export AFFINITY_BENCHMARK_BUDGET_USD=5
```

See `.env.example`.

## Commands

```bash
python tools/affinity_vast.py offers
python tools/affinity_vast.py benchmark          # local GPU qualification
python tools/affinity_vast.py plan --budget 100 --target-hours 168 --max-gpus 20
python tools/affinity_vast.py launch --yes       # prints preview economics; requires --yes
python tools/affinity_vast.py status
python tools/affinity_vast.py stop               # destroys instances (disk-safe)
python tools/affinity_vast.py destroy-all --yes
python tools/affinity_vast.py claim-server       # HTTP claims for local+Vast
```

`plan` never launches. `launch` prints qualified GPU, fleet $/hour, TPS, chunks/hour, remaining chunks, ETA, and cost before provisioning.

## Claim coordination (local + Vast)

Do **not** use DynamoDB.

- Single host / shared FS: `S7_CLAIM_BACKEND=local` (default)
- Multi-host (Vast workers + local ROCm): run claim server on the controller:

```bash
python tools/affinity_vast.py claim-server --port 8787
# workers:
export S7_CLAIM_BACKEND=http
export S7_CLAIM_HTTP=http://CONTROLLER:8787
```

Interrupted Vast workers must not mark chunks complete without `affinities_core_czyx.npy` on the controller (enforced in `release_local`).

## One worker per GPU

Multiple full MNet processes on one GPU are not the production path unless a future receipt proves better **$/chunk**.

## Receipts

Written under `experiments/phase6e/VAST-AFFINITY-BENCH-001/`:

- `VAST_GPU_OFFERS.json`
- `VAST_GPU_BENCHMARKS.json`
- `VAST_COST_MODEL.json`
- `VAST_SELECTED_GPU.json`
- `VAST_FLEET_PLAN.json`
- `budget_ledger.json`

## Equivalence (unchanged)

`hyperdrain.equivalence.compare_volumes`:
- `abs_tol = 1e-3`
- decision threshold `0.5`
- `max_decision_disagree_frac = 1e-4`

GATE_A–E remain frozen. Throughput work does not reopen model qualification.
