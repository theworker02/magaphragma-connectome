#!/usr/bin/env bash
# Emergency disk hygiene on existing 3060 Ti (keep instance alive).
set -euo pipefail
KEY="${VAST_KEY:-/mnt/c/Users/matth/.ssh/vast_affinity}"
KEY_USE=/tmp/vast_disk_$$
cp "$KEY" "$KEY_USE" && chmod 600 "$KEY_USE"
HOST=92.190.14.191
PORT=28225
SSH=(ssh -p "$PORT" -i "$KEY_USE" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -o ConnectTimeout=20 -o BatchMode=yes)
"${SSH[@]}" "root@$HOST" bash -s <<'REMOTE'
set -euo pipefail
echo "=== before ==="
df -h /workspace | tail -1
# Offload-ready chunks already pulled locally should be deleted; also clear caches/tmp
du -sh /workspace/s7-out/chunks 2>/dev/null || true
# Keep queue_state; remove completed chunk dirs older than fresh and temp junk
find /workspace/s7-out/chunks -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -5 || true
# Aggressive: remove affinities already complete if local offload may have them —
# Prefer delete only tmp/stage/cache, leave COMPLETE dirs for offload script.
rm -rf /tmp/s7-accum/* /tmp/s7-fast-stage/* /root/.cache/pip /workspace/.cache 2>/dev/null || true
# Remove incomplete / empty chunk dirs
find /workspace/s7-out/chunks -mindepth 1 -maxdepth 1 -type d -empty -delete 2>/dev/null || true
# If still tight, delete SEG_PENDING/COMPLETE chunk payloads (offload should have pulled;
# worker recreates queue claims). Keep queue_state.json and claims.
if [[ $(df -Pk /workspace | awk 'NR==2{print $4}') -lt 5000000 ]]; then
  echo "LOW_DISK — pruning completed chunk dirs"
  find /workspace/s7-out/chunks -mindepth 1 -maxdepth 1 -type d | while read -r d; do
    if [[ -f "$d/affinities_core_czyx.npy" ]] || [[ -f "$d/DONE" ]]; then
      rm -rf "$d"
    fi
  done
fi
echo "=== after ==="
df -h /workspace | tail -1
# Restart worker on fleet shard 0/10 with fp16 memmap
pkill -f 'run_affinity_fullvol_s7_fast_worker' 2>/dev/null || true
pkill -f 'run_vast_simple.sh' 2>/dev/null || true
pkill -f 'vast_worker_loop' 2>/dev/null || true
sleep 2
mkdir -p /tmp/s7-accum /tmp/s7-fast-stage /tmp/s7-logs /workspace/s7-out
export S7_SHARD=0/10
export S7_WORKER_ID=vast-3060ti
export S7_TILE_BATCH=8
export S7_ACCUM_DTYPE=float16
export S7_ACCUM_MEMMAP_DIR=/tmp/s7-accum
export S7_STREAM_SAVE=1
cd /workspace/magaphragma-connectome
nohup bash -c 'while true; do
  S7_SHARD=0/10 S7_WORKER_ID=vast-3060ti S7_TILE_BATCH=8 bash tools/run_vast_simple.sh >>/tmp/s7-logs/worker.log 2>&1
  echo "[$(date -u +%FT%TZ)] worker exited; respawn" >>/tmp/s7-logs/worker.log
  sleep 20
done' >/tmp/s7-logs/loop.log 2>&1 &
echo RESTARTED_PID=$!
sleep 4
tail -n 20 /tmp/s7-logs/worker.log || tail -n 20 /workspace/s7-out/worker.log || true
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader || true
REMOTE
rm -f "$KEY_USE"
