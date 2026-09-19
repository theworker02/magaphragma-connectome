#!/usr/bin/env bash
# Windows/Debian-friendly Vast→local offload (no Ubuntu-24.04 required).
# Uses scp instead of rsync when needed. Defaults match Host vast-affinity.
set -euo pipefail
HOST="${VAST_HOST:-92.190.14.191}"
PORT="${VAST_PORT:-28225}"
KEY="${VAST_KEY:-/mnt/c/Users/matth/.ssh/vast_affinity}"
LOCAL_CHUNKS="${LOCAL_CHUNKS:-/mnt/c/Users/matth/s7-affinity-out/chunks}"
REMOTE_CHUNKS="${REMOTE_CHUNKS:-/workspace/s7-out/chunks}"
LOG="${LOG:-/mnt/c/Users/matth/s7-affinity-out/vast_offload_win.log}"
INTERVAL_SEC="${INTERVAL_SEC:-120}"
KEEP_REMOTE="${KEEP_REMOTE:-0}"
LOW_DISK_KB="${LOW_DISK_KB:-2500000}"

LOOP=0
[[ "${1:-}" == "--loop" ]] && LOOP=1

mkdir -p "$LOCAL_CHUNKS" "$(dirname "$LOG")"
KEY_USE=/tmp/vast_affinity_offload_key
cp "$KEY" "$KEY_USE" && chmod 600 "$KEY_USE"

SSH=(ssh -p "$PORT" -i "$KEY_USE" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20)

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

free_kb() {
  "${SSH[@]}" "root@$HOST" "df -Pk / | awk 'NR==2{print \$4}'" | tr -d '\r'
}

list_completed() {
  "${SSH[@]}" "root@$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
CHUNKS=/workspace/s7-out/chunks
[[ -d "$CHUNKS" ]] || exit 0
for d in "$CHUNKS"/MV-CHUNK-*; do
  [[ -d "$d" ]] || continue
  cid=$(basename "$d")
  [[ -f "$d/receipt.json" ]] || continue
  [[ -f "$d/affinities_core_czyx.npy" ]] || continue
  [[ -f "$d/boundaries_core.tif" ]] || continue
  asz=$(stat -c%s "$d/affinities_core_czyx.npy" 2>/dev/null || echo 0)
  [[ "$asz" -gt 100000000 ]] || continue
  echo "$cid"
done
REMOTE
}

offload_once() {
  local avail cid
  avail=$(free_kb)
  log "free_kb=${avail} keep_remote=${KEEP_REMOTE}"
  local done_list
  done_list=$(list_completed | tr -d '\r' || true)
  if [[ -z "${done_list//[[:space:]]/}" ]]; then
    log "no completed chunks on Vast"
    return 0
  fi
  log "completed_on_vast=$(printf '%s\n' "$done_list" | grep -c . || true)"
  while read -r cid; do
    [[ -n "$cid" ]] || continue
    log "scp $cid → $LOCAL_CHUNKS/$cid"
    mkdir -p "$LOCAL_CHUNKS/$cid"
    if ! scp -P "$PORT" -i "$KEY_USE" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -r \
      "root@$HOST:$REMOTE_CHUNKS/$cid/affinities_core_czyx.npy" \
      "root@$HOST:$REMOTE_CHUNKS/$cid/boundaries_core.tif" \
      "root@$HOST:$REMOTE_CHUNKS/$cid/receipt.json" \
      "$LOCAL_CHUNKS/$cid/"; then
      log "ERROR scp failed $cid"
      continue
    fi
    local_sz=$(stat -c%s "$LOCAL_CHUNKS/$cid/affinities_core_czyx.npy")
    remote_sz=$("${SSH[@]}" "root@$HOST" "stat -c%s $REMOTE_CHUNKS/$cid/affinities_core_czyx.npy" | tr -d '\r')
    if [[ "$local_sz" != "$remote_sz" ]]; then
      log "ERROR size mismatch $cid local=$local_sz remote=$remote_sz"
      continue
    fi
    if [[ "$KEEP_REMOTE" == "1" ]]; then
      log "verified $cid KEEP_REMOTE"
      continue
    fi
    log "verified $cid — deleting remote"
    "${SSH[@]}" "root@$HOST" "rm -rf $REMOTE_CHUNKS/$cid" || log "ERROR delete $cid"
  done <<<"$done_list"
  avail=$(free_kb)
  log "done free_kb=$avail"
  if [[ "$avail" -lt "$LOW_DISK_KB" ]]; then
    log "WARN still below LOW_DISK"
  fi
}

if [[ "$LOOP" -eq 1 ]]; then
  log "offload loop start interval=${INTERVAL_SEC}s"
  while true; do
    offload_once || log "WARN offload_once rc=$?"
    sleep "$INTERVAL_SEC"
  done
else
  offload_once
fi
