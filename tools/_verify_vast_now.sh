#!/usr/bin/env bash
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new root@ssh3.vast.ai bash -s <<'REMOTE'
set +e
echo "=== processes ==="
ps aux | grep -E 'run_affinity_fullvol|run_vast_simple' | grep -v grep
echo "=== nvidia ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
echo "=== log ==="
tail -n 40 /workspace/s7-out/worker.log
REMOTE
