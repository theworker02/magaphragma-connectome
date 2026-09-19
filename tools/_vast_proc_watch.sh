#!/usr/bin/env bash
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new root@ssh3.vast.ai bash -s <<'REMOTE'
set +e
PID=$(pgrep -f 'run_affinity_fullvol_s7_fast_worker.py' | head -1)
echo "pid=$PID"
if [[ -n "$PID" ]]; then
  ps -p "$PID" -o pid,etime,pcpu,pmem,stat,wchan:20,cmd
  echo "=== status snippet ==="
  grep -E 'State:|VmRSS:|VmSwap:|Threads:' /proc/$PID/status
fi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
free -h
echo "=== log tail ==="
tail -n 40 /workspace/s7-out/worker.log
sleep 90
echo "=== after 90s ==="
PID=$(pgrep -f 'run_affinity_fullvol_s7_fast_worker.py' | head -1)
echo "pid=$PID"
if [[ -n "$PID" ]]; then
  ps -p "$PID" -o pid,etime,pcpu,pmem,stat,wchan:20,cmd
  grep -E 'State:|VmRSS:|VmSwap:' /proc/$PID/status
else
  echo DEAD
fi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
tail -n 50 /workspace/s7-out/worker.log
REMOTE
