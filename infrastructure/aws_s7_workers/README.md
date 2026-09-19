# Magaphragma S7 AWS GPU fleet — HISTORICAL / DECOMMISSIONED

> **Status:** DECOMMISSIONED for Affinity production (2026-09-18).
> Production cloud provider is **Vast.ai**. See `docs/VAST_AFFINITY.md` and `docs/AWS_DECOMMISSION.md`.

This folder is retained as **historical infrastructure evidence** only.
Do not deploy `s7-gpu-fleet.yaml`. Do not scale the ASG. Do not seed DynamoDB.

Active path:
```bash
python tools/affinity_vast.py offers
python tools/affinity_vast.py plan --budget 100 --target-hours 168
```

Local ROCm worker remains supported via `tools/run_affinity_fullvol_s7_fast_worker.py`.
