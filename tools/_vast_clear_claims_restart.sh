#!/usr/bin/env bash
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new root@ssh3.vast.ai bash -s <<'REMOTE'
set -euo pipefail
echo "=== clear stale claims ==="
ls /workspace/s7-out/claims 2>/dev/null | wc -l
rm -f /workspace/s7-out/claims/*
# reset all RUNNING without affinity files
python3 - <<'PY'
import json
from pathlib import Path
out = Path("/workspace/s7-out")
state = json.loads((out/"queue_state.json").read_text())
n=0
for cid, st in list(state.get("status_by_id", {}).items()):
    if st in ("RUNNING", "FAILED") and not (out/"chunks"/cid/"affinities_core_czyx.npy").exists():
        state["status_by_id"][cid] = "NOT_STARTED"
        n += 1
(out/"queue_state.json").write_text(json.dumps(state))
print("reset_to_not_started", n)
PY
# soft-restart worker process only (loop stays)
pkill -f 'run_affinity_fullvol_s7_fast_worker.py' || true
sleep 3
pgrep -af 'vast_worker_loop|run_affinity' || echo 'loop dead — restarting'
if ! pgrep -f vast_worker_loop >/dev/null; then
  nohup bash /workspace/s7-out/vast_worker_loop.sh >/workspace/s7-out/loop.out 2>&1 &
  echo restarted_loop=$!
fi
sleep 20
echo "=== status ==="
pgrep -af 'vast_worker_loop|run_affinity' || echo DEAD
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
tail -n 50 /workspace/s7-out/worker.log
REMOTE
