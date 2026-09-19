# Handoff — unfinished, takeover-ready

**Status:** Public handoff of an unfinished but substantial connectome research codebase. Most of the heavy lifting (evidence infrastructure, Affinity S7 FAST path, Vast/local hybrid fleet tooling, AWS decommission) is done. The full-volume affinity run is **not** finished.

This is not a finished biological connectome. There is no releasable Vigilia reconstruction. Takeover means continuing the **technical** Affinity S7 campaign and the evidence-first programme under the same scientific guardrails.

If you only read one other file after this, start with [README.md](README.md) and [START_HERE.md](START_HERE.md).

---

## What's done

| Area | State |
| --- | --- |
| Evidence-first core (`src/mvconnectome/`) | Substantial — typed records, provenance, gated promotion |
| Phase 5C production domain | **28,798-chunk** resumable queue planned over the reported parent extent |
| Affinity training / qualification funnel | Parallel funnel + GATE_A–E contracts; `checkpoint-step10.pt` path established |
| Affinity S7 FAST inference | Workers, claim coordination, durable commit, progress tooling |
| Vast.ai production cloud path | `tools/affinity_vast.py`, budgets, offer plan/launch/stop |
| Local AMD ROCm worker | First-class free concurrent worker on the same claim queue |
| Hybrid fleet docs | [docs/HYBRID_FLEET.md](docs/HYBRID_FLEET.md) |
| AWS Affinity path | **DECOMMISSIONED** — [docs/AWS_DECOMMISSION.md](docs/AWS_DECOMMISSION.md) |
| S6 authorization for S7 | Recorded APPROVED (`experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json`) |
| Contracts / receipts (JSON) | Large set under `experiments/phase6e/*.json` (kept in git) |

Last recorded S7 snapshot on the originating machine (indicative only; your checkout will not include chunk binaries):

- Total chunks: **28,798**
- Completed (local status): on the order of **~200–240** affinity chunks
- Remaining: **~28.5k**
- Cloud spend recorded in fleet status: **$0** (do not assume this is still true on any live Vast account)

---

## What's left (takeover checklist)

1. **Obtain the Affinity checkpoint separately**  
   Path used by fleet docs:  
   `experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt`  
   (~460 MB). **Not in this GitHub upload** (exceeds practical git limits). Ask the previous maintainer or regenerate via the funnel contracts if you have the training tree.

2. **Fleet bootstrap / auth push to Vast**  
   Wire `VAST_API_KEY`, SSH/rsync of lean tree + checkpoint, claim HTTP reachability (Tailscale / reverse tunnel). See [docs/VAST_AFFINITY.md](docs/VAST_AFFINITY.md) and [docs/HYBRID_FLEET.md](docs/HYBRID_FLEET.md).

3. **WSL ROCm recovery**  
   Local free worker expects a working ROCm torch env (historically `/opt/venvs/unlearning-rocm`). Re-verify `torch.cuda.is_available()` / HIP before counting on local TPS.

4. **Scale to finish ~28k chunks**  
   Run claim coordinator + local and/or Vast workers under hard budgets (`AFFINITY_CLOUD_BUDGET_USD=160`). Prefer `tools/affinity_vast.py plan` before any `launch --yes`.

5. **Merge / durable-collect Vast outputs**  
   Chunks must land as `affinities_core_czyx.npy` on the controller before claim release marks complete. Merge remote durable uploads; do not mark done on empty claims.

6. **Post-affinity programme (not opened by finishing S7 alone)**  
   S8 candidate seg / proofreading, synapse/connectivity, CATMAID↔DVID registration (still blocked), rights for any redistribution — see research ledgers.

**Do not** resume a paid Vast fleet unless you intentionally accept spend. Hard budget gates exist for a reason.

---

## How to run (key commands)

### Secrets

```bash
cp .env.example .env
# Set VAST_API_KEY locally — never commit .env
```

### Vast controller CLI

```bash
python tools/affinity_vast.py offers
python tools/affinity_vast.py plan --budget 100 --target-hours 168 --max-gpus 20
# launch spends money — requires --yes and passes budget checks
python tools/affinity_vast.py status
python tools/affinity_vast.py stop
python tools/affinity_vast.py claim-server --port 8787
```

### Local / hybrid workers

See [docs/HYBRID_FLEET.md](docs/HYBRID_FLEET.md). Short form:

```bash
# Claim coordinator (controller)
python -u tools/cloud/claim_http.py

# Local ROCm worker (WSL example)
export S7_WORKER_ID=local-rx7800xt-01
export S7_CLAIM_BACKEND=local
python -u tools/run_affinity_fullvol_s7_fast_worker.py \
  --claim --loop --max-chunks 64 --batch-size 16 \
  --contract FAST_003 --infer-backend eager

# Progress
python tools/affinity_fleet.py status --claim-http http://127.0.0.1:8787
python tools/s7_progress_snapshot.py
```

Simple Vast worker bootstrap (after rsync + checkpoint):

```bash
bash tools/bootstrap_vast_affinity.sh
# or
bash tools/run_vast_simple.sh
```

### Budgets

| Env | Default | Meaning |
| --- | --- | --- |
| `AFFINITY_CLOUD_BUDGET_USD` | 160 | Hard ceiling |
| `AFFINITY_TARGET_BUDGET_USD` | 100 | Preferred target |
| `AFFINITY_BENCHMARK_BUDGET_USD` | 5 | Qualification only |

Details: [docs/AFFINITY_COST_MODEL.md](docs/AFFINITY_COST_MODEL.md).

---

## What this GitHub repo deliberately excludes

To stay under GitHub limits and avoid leaking secrets / raw EM:

- `.env` and API keys
- Multi-GB `experiments/**/chunks/` affinity dumps (`.npy`)
- Model checkpoints (`.pt` / `.pth`)
- `datasets/cache/`, `local_research_build/`, `derived/`
- Nested third-party clones (`segneuron`, `ffn`, …) and their virtualenvs — see [third_party/README.md](third_party/README.md)

**Included:** source (`src/`), `tools/`, `tests/`, `docs/`, contracts/receipts JSON, lean configs, and documentation.

If you need the originating machine's chunk tree or checkpoint, obtain them out-of-band — do not expect `git clone` alone to resume mid-volume inference.

---

## Docs map

| Doc | Purpose |
| --- | --- |
| [docs/VAST_AFFINITY.md](docs/VAST_AFFINITY.md) | Vast production path + budgets |
| [docs/HYBRID_FLEET.md](docs/HYBRID_FLEET.md) | Local ROCm + Vast shared queue |
| [docs/AWS_DECOMMISSION.md](docs/AWS_DECOMMISSION.md) | AWS Affinity retired |
| [reports/CONNECTOME_STATUS.md](reports/CONNECTOME_STATUS.md) | Authoritative local inventory |
| [research/CAPABILITY_MATRIX.md](research/CAPABILITY_MATRIX.md) | Capability ledger |
| [research/BLOCKER_RESOLUTION.md](research/BLOCKER_RESOLUTION.md) | Blockers |

---

## Scientific non-negotiables (carry forward)

1. Machine affinities/segments are **not** biology until append-only human review + integrity gates say so.
2. No invented neurons, synapses, or connectivity for demos.
3. CATMAID ↔ DVID physical registration remains **blocked** until independently supported.
4. Public HTTP availability ≠ redistribution rights.

---

## Contact / sponsorship

GitHub: [@theworker02](https://github.com/theworker02)  
Optional sponsorship: [SUPPORT.md](SUPPORT.md) / funding links on the repo page.
