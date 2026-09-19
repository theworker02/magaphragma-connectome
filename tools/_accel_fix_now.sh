#!/bin/bash
# Emergency restore: amdgpu + correct-host offload + local worker.
set -euo pipefail
export HOME=/home/research-runner
export USER=research-runner
ROOT=/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
cd "$ROOT"

echo "=== amdgpu ==="
sudo modprobe amdgpu 2>&1 || true
sleep 2
lsmod | grep amdgpu || echo NO_AMDGPU
/opt/rocm/bin/rocm-smi 2>&1 | head -20 || true
/opt/venvs/unlearning-rocm/bin/python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)" 2>&1

if ! grep -q 'HOME=/home/research-runner' /home/research-runner/.bashrc 2>/dev/null; then
  echo 'export HOME=/home/research-runner' >> /home/research-runner/.bashrc
fi

cp /mnt/c/Users/matth/.ssh/vast_affinity /tmp/vast_affinity
chmod 600 /tmp/vast_affinity
export VAST_HOST=92.190.14.191
export VAST_PORT=28225
export VAST_KEY=/tmp/vast_affinity
export LOCAL_CHUNKS=/home/research-runner/s7-affinity-out/chunks
export LOG=/home/research-runner/s7-affinity-out/vast_offload.log
mkdir -p "$LOCAL_CHUNKS" /home/research-runner/s7-affinity-out

pkill -f 'offload_vast_s7_chunks.sh --loop' 2>/dev/null || true
sleep 1
echo "=== one-shot offload ==="
bash tools/offload_vast_s7_chunks.sh 2>&1 | tee -a "$LOG" | tail -80
echo "=== start offload loop ==="
nohup env VAST_HOST="$VAST_HOST" VAST_PORT="$VAST_PORT" VAST_KEY="$VAST_KEY" HOME="$HOME" LOCAL_CHUNKS="$LOCAL_CHUNKS" LOG="$LOG" \
  bash tools/offload_vast_s7_chunks.sh --loop >/dev/null 2>&1 &
echo $! > /home/research-runner/s7-affinity-out/vast_offload.pid
echo "offload_pid=$(cat /home/research-runner/s7-affinity-out/vast_offload.pid)"
pgrep -af offload_vast_s7 || true

echo "=== start local worker shard 0/2 ==="
pkill -f 'run_affinity_fullvol_s7_fast_worker.py' 2>/dev/null || true
sleep 2
mkdir -p /tmp/s7-fast-stage
nohup bash tools/run_local_simple.sh >> /home/research-runner/s7-affinity-out/local_worker.log 2>&1 &
echo "local_worker_pid=$!"
sleep 5
pgrep -af 'run_affinity_fullvol_s7_fast_worker' || echo 'local worker not yet visible'
tail -30 /home/research-runner/s7-affinity-out/local_worker.log || true
df -h /
find /home/research-runner/s7-affinity-out -name affinities_core_czyx.npy | wc -l
