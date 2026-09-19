#!/bin/bash
export HOME=/home/research-runner
echo alive
id
lsmod | grep amdgpu || echo NO_AMDGPU
ls /dev/dri 2>&1 || true
ls /dev/kfd 2>&1 || true
df -h / | tail -1
find /home/research-runner/s7-affinity-out -name affinities_core_czyx.npy | wc -l
# offload with correct host
cp /mnt/c/Users/matth/.ssh/vast_affinity /tmp/vast_affinity
chmod 600 /tmp/vast_affinity
cd /mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
export VAST_HOST=92.190.14.191
export VAST_PORT=28225
export VAST_KEY=/tmp/vast_affinity
export LOCAL_CHUNKS=/home/research-runner/s7-affinity-out/chunks
export LOG=/home/research-runner/s7-affinity-out/vast_offload.log
mkdir -p "$LOCAL_CHUNKS"
echo "=== oneshot offload ==="
bash tools/offload_vast_s7_chunks.sh
echo "=== done offload rc=$? ==="
df -h / | tail -1
