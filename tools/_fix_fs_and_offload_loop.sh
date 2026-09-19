#!/bin/bash
export HOME=/home/research-runner
echo "=== fs check ==="
mount | grep ' / '
dmesg 2>/dev/null | tail -20 || true
touch /home/research-runner/s7-affinity-out/.write_test 2>&1 && echo writable || echo READONLY
# remount rw if needed
sudo mount -o remount,rw / 2>&1 || true
touch /home/research-runner/s7-affinity-out/.write_test 2>&1 && echo writable_after || echo still_ro
find /home/research-runner/s7-affinity-out -name affinities_core_czyx.npy | wc -l
# start offload loop with correct host
cp /mnt/c/Users/matth/.ssh/vast_affinity /tmp/vast_affinity
chmod 600 /tmp/vast_affinity
cd /mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
export VAST_HOST=92.190.14.191 VAST_PORT=28225 VAST_KEY=/tmp/vast_affinity
export LOCAL_CHUNKS=/home/research-runner/s7-affinity-out/chunks
export LOG=/home/research-runner/s7-affinity-out/vast_offload.log
pkill -f 'offload_vast_s7_chunks.sh --loop' 2>/dev/null || true
bash tools/offload_vast_s7_chunks.sh || true
nohup env VAST_HOST=$VAST_HOST VAST_PORT=$VAST_PORT VAST_KEY=$VAST_KEY HOME=$HOME LOCAL_CHUNKS=$LOCAL_CHUNKS LOG=$LOG INTERVAL_SEC=120 \
  bash tools/offload_vast_s7_chunks.sh --loop >/dev/null 2>&1 &
echo $! > /home/research-runner/s7-affinity-out/vast_offload.pid
echo offload_pid=$(cat /home/research-runner/s7-affinity-out/vast_offload.pid)
pgrep -af offload_vast || true
# try enable GPU via mesa/dxg
ls /usr/lib/wsl/lib 2>/dev/null | head
ls /dev/dxg 2>&1 || true
