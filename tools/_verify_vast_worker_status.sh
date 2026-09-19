#!/bin/bash
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new root@ssh3.vast.ai bash -s <<'REMOTE'
set +e
echo "==== PROCESS ===="
pgrep -af 'run_affinity_fullvol_s7_fast_worker|run_vast_simple' || echo "(no matching process)"
echo
echo "==== QUEUE_STATE ===="
ls -la /workspace/s7-out/queue_state.json 2>&1
echo
echo "==== CHUNKS ===="
ls /workspace/s7-out/chunks 2>/dev/null | head -20
echo "count=$(ls /workspace/s7-out/chunks 2>/dev/null | wc -l)"
echo
echo "==== NVIDIA ===="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
echo
echo "==== LOG TAIL ===="
tail -n 100 /workspace/s7-out/worker.log
echo
echo "==== KEY LINES ===="
grep -E 'FileNotFoundError|Traceback|created queue_state|shard=|claim|tile|chunk|Error' /workspace/s7-out/worker.log | tail -n 50
REMOTE
