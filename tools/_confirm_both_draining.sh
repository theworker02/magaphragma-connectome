#!/usr/bin/env bash
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new root@ssh3.vast.ai bash -s <<'REMOTE'
set +e
# try grow swap headroom with a second file if disk allows
if [[ ! -f /swapfile2 ]]; then
  echo "=== adding /swapfile2 8G ==="
  fallocate -l 8G /swapfile2 && chmod 600 /swapfile2 && mkswap /swapfile2 && swapon /swapfile2 || echo "swap2 failed"
fi
free -h
echo "=== t0 ==="
date -u
pgrep -af 'run_affinity_fullvol|vast_worker_loop' || echo DEAD
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
tail -n 20 /workspace/s7-out/worker.log
sleep 45
echo "=== t1 (+45s) ==="
date -u
pgrep -af 'run_affinity_fullvol|vast_worker_loop' || echo DEAD
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
free -h
tail -n 30 /workspace/s7-out/worker.log
REMOTE
echo
echo "=== LOCAL ==="
pgrep -af run_affinity_fullvol_s7_fast_worker || echo none
tail -n 15 /mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome/experiments/phase6e/AFFINITY-FULLVOL-S7-001/local_worker.log
