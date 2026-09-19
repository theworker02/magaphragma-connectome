#!/usr/bin/env bash
# Start (or restart) the WSL background Vast→local chunk offload loop.
# Does not touch the Vast inference worker or local shard 0/2.
set -euo pipefail

ROOT="${ROOT:-/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome}"
SCRIPT="$ROOT/tools/offload_vast_s7_chunks.sh"
LOG="${LOG:-$HOME/s7-affinity-out/vast_offload.log}"
PIDFILE="${PIDFILE:-$HOME/s7-affinity-out/vast_offload.pid}"
INTERVAL_SEC="${INTERVAL_SEC:-180}"

mkdir -p "$(dirname "$LOG")"
chmod +x "$SCRIPT" 2>/dev/null || true

# Ensure key is usable from WSL
if [[ ! -f /tmp/vast_affinity ]]; then
  cp /mnt/c/Users/matth/.ssh/vast_affinity /tmp/vast_affinity
fi
chmod 600 /tmp/vast_affinity

if [[ -f "$PIDFILE" ]]; then
  old=$(cat "$PIDFILE" || true)
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    # Only kill if it looks like our offload loop
    if ps -p "$old" -o args= 2>/dev/null | grep -q offload_vast_s7_chunks; then
      echo "stopping previous offload pid=$old"
      kill "$old" 2>/dev/null || true
      sleep 1
    fi
  fi
fi
# Clear any orphaned loops (best-effort)
pkill -f 'offload_vast_s7_chunks.sh --loop' 2>/dev/null || true
sleep 2

export VAST_KEY=/tmp/vast_affinity
# Keep in sync with ~/.ssh/config Host vast-affinity
export VAST_HOST="${VAST_HOST:-92.190.14.191}"
export VAST_PORT="${VAST_PORT:-28225}"
export INTERVAL_SEC
export KEEP_REMOTE="${KEEP_REMOTE:-0}"
export HOME="${HOME:-/home/research-runner}"
export LOCAL_CHUNKS="${LOCAL_CHUNKS:-$HOME/s7-affinity-out/chunks}"
export LOG="${LOG:-$HOME/s7-affinity-out/vast_offload.log}"

# One immediate pass, then start the background loop (avoids overlapping first ticks)
bash "$SCRIPT" || true

nohup bash "$SCRIPT" --loop >/dev/null 2>&1 &
echo $! >"$PIDFILE"
echo "started offload loop pid=$(cat "$PIDFILE") interval=${INTERVAL_SEC}s log=$LOG"

echo "=== status ==="
ps -p "$(cat "$PIDFILE")" -o pid,etime,cmd || echo "loop not running"
pgrep -af 'offload_vast_s7_chunks.sh --loop' || true
df -h / | head -2
tail -n 15 "$LOG" || true
