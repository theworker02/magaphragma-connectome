#!/usr/bin/env bash
# Restart local Affinity S7 FAST worker on shard 0/2 (RX 7800 XT / ROCm).
set -euo pipefail
REPO=/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
cd "$REPO"
PY=/opt/venvs/unlearning-rocm/bin/python
LOG="$REPO/experiments/phase6e/AFFINITY-FULLVOL-S7-001/local_worker.log"
mkdir -p "$(dirname "$LOG")"

echo "=== stopping old workers ==="
pkill -f 'run_affinity_fullvol_s7_fast_worker.py' 2>/dev/null || true
sleep 2
# ensure dead
if pgrep -f 'run_affinity_fullvol_s7_fast_worker.py' >/dev/null; then
  pkill -9 -f 'run_affinity_fullvol_s7_fast_worker.py' 2>/dev/null || true
  sleep 1
fi

echo "=== torch/rocm probe ==="
"$PY" - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("cuda_is_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY

echo "=== starting shard 0/2 ==="
export S7_WORKER_ID=local-rx7800xt-01
export S7_CLAIM_BACKEND=local
export S7_DURABLE_MODE=local
export S7_TILE_BATCH=16
export S7_INFER_BACKEND=eager
# default S7_OUT = experiments/phase6e/AFFINITY-FULLVOL-S7-001 (separate from Vast)

nohup bash tools/run_local_simple.sh > "$LOG" 2>&1 &
echo "pid=$!"
sleep 10
echo "=== processes ==="
pgrep -af 'run_affinity_fullvol_s7_fast_worker' || echo "(none)"
echo "=== log ==="
tail -n 40 "$LOG" || true
