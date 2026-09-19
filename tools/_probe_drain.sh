#!/usr/bin/env bash
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new root@ssh3.vast.ai bash -s <<'REMOTE'
set +e
echo "=== t1 ==="
date -u
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv
tail -n 8 /workspace/s7-out/worker.log
sleep 20
echo "=== t2 (+20s) ==="
date -u
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv
tail -n 12 /workspace/s7-out/worker.log
ps -p 11067 -o pid,etime,pcpu,pmem,cmd
ls -la /workspace/s7-out/chunks | head -10
REMOTE
echo
echo "=== LOCAL LOG ==="
tail -n 20 /mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome/experiments/phase6e/AFFINITY-FULLVOL-S7-001/local_worker.log
echo "=== LOCAL PROC ==="
pgrep -af run_affinity_fullvol_s7_fast_worker || echo none
