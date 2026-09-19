#!/bin/bash
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
SSH=(ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
"${SSH[@]}" root@ssh3.vast.ai bash -s <<'REMOTE'
set -euo pipefail
echo "=== paths ==="
ls -la /workspace/magaphragma-connectome/local_research_build/phase5c-production/chunks.json 2>&1 || true
ls -la /workspace/s7-out 2>&1 || true
ls -la /workspace/magaphragma-connectome/tools/s7_chunk_claim.py
echo "=== claim head ==="
head -25 /workspace/magaphragma-connectome/tools/s7_chunk_claim.py
echo "=== starting worker (nohup) ==="
pkill -f 'run_affinity_fullvol_s7_fast_worker' 2>/dev/null || true
sleep 1
cd /workspace/magaphragma-connectome
nohup bash tools/run_vast_simple.sh > /workspace/s7-out/worker.log 2>&1 &
echo "pid=$!"
sleep 8
echo "=== log ==="
tail -n 40 /workspace/s7-out/worker.log 2>&1 || true
REMOTE
