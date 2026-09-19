#!/usr/bin/env bash
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new root@ssh3.vast.ai bash -s <<'REMOTE'
set +e
echo "=== alive? ==="
pgrep -af run_affinity_fullvol || echo DEAD
echo "=== full worker.log ==="
wc -l /workspace/s7-out/worker.log
cat /workspace/s7-out/worker.log
echo "=== dmesg OOM/CUDA ==="
dmesg -T 2>/dev/null | tail -n 40
echo "=== nvidia compute apps ==="
nvidia-smi
echo "=== free -h ==="
free -h
echo "=== disk ==="
df -h /workspace /tmp
REMOTE
