#!/usr/bin/env bash
# Start local Affinity controller + RX 7800 XT worker for hybrid fleet.
# Run inside WSL.
set -euo pipefail
ROOT="/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome"
cd "$ROOT"
PY="/opt/venvs/unlearning-rocm/bin/python"
mkdir -p /tmp/s7-fleet

# 1) Claim coordinator (shared queue)
if ! curl -sf http://127.0.0.1:8787/health >/dev/null 2>&1; then
  nohup "$PY" -u tools/cloud/claim_http.py >>/tmp/s7-fleet/claim_http.log 2>&1 &
  echo "claim_http pid=$!"
  sleep 1
fi
curl -s http://127.0.0.1:8787/health
echo

# 2) Conservative stale reclaim (only age-expired RUNNING without affinities)
"$PY" tools/migrate_s7_queue_for_fleet.py --max-age-s 7200

# 3) Local worker
export S7_WORKER_ID=local-rx7800xt-01
export S7_CLAIM_BACKEND=local
export S7_DURABLE_MODE=local
export S7_TILE_BATCH=16
export S7_INFER_BACKEND=eager
export S7_CLAIM_HTTP=http://127.0.0.1:8787

# Register for fleet status (optional)
"$PY" - <<'PY' || true
import os
from cloud.claim_http import register_worker_http
register_worker_http(os.environ["S7_CLAIM_HTTP"], {
    "worker_id": "local-rx7800xt-01",
    "gpu": "AMD Radeon RX 7800 XT",
    "runtime": "ROCm",
    "batch": 16,
    "backend": "eager",
    "dph_usd": 0.0,
})
print("registered local-rx7800xt-01")
PY

if pgrep -f "run_affinity_fullvol_s7_fast_worker.py" >/dev/null 2>&1; then
  echo "local worker already running — restart manually if you need worker_id=local-rx7800xt-01"
else
  nohup "$PY" -u tools/run_affinity_fullvol_s7_fast_worker.py \
    --claim --loop --max-chunks 64 --batch-size 16 \
    --contract FAST_003 --infer-backend eager \
    >>experiments/phase6e/AFFINITY-FULLVOL-S7-001/worker_fast003.log 2>&1 &
  echo "worker pid=$!"
fi

echo "Status:"
"$PY" tools/affinity_fleet.py status --claim-http http://127.0.0.1:8787
echo
echo "Expose coordinator to Vast (pick one):"
echo "  - Tailscale: use this machine's Tailscale IP as CONTROLLER"
echo "  - SSH reverse tunnel from this PC: ssh -N -R 8787:127.0.0.1:8787 root@VAST_HOST -p VAST_PORT"
