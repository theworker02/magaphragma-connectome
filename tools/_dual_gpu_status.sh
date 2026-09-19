#!/usr/bin/env bash
# One-shot dual-GPU status for Affinity S7 FAST (local shard 0/2 + Vast shard 1/2).
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
SSH=(ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)

echo "========== VAST =========="
"${SSH[@]}" root@ssh3.vast.ai bash -s <<'REMOTE'
set +e
echo "=== processes ==="
ps aux | grep -E 'run_affinity_fullvol|run_vast_simple' | grep -v grep
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
echo "=== worker.log (tail 40) ==="
tail -n 40 /workspace/s7-out/worker.log 2>&1
echo "=== chunks count ==="
ls /workspace/s7-out/chunks 2>/dev/null | wc -l
echo "=== queue_state ==="
ls -la /workspace/s7-out/queue_state.json 2>&1
REMOTE

echo
echo "========== LOCAL =========="
echo "=== processes ==="
pgrep -af 'run_affinity_fullvol_s7_fast_worker' || echo "(none)"
echo "=== rocm-smi ==="
(command -v rocm-smi >/dev/null && rocm-smi --showuse) || (/opt/rocm/bin/rocm-smi --showuse 2>/dev/null) || echo "rocm-smi unavailable"
