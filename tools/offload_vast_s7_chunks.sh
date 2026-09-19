#!/usr/bin/env bash
# Pull completed Vast Affinity S7 chunks → WSL durable store, verify bytes, free Vast disk.
#
# Safe for concurrent inference: only touches chunks with receipt.json present
# (worker writes receipt last). Never deletes mid-write; never touches local shard 0/2.
#
# Usage (WSL):
#   bash tools/offload_vast_s7_chunks.sh              # one shot
#   bash tools/offload_vast_s7_chunks.sh --loop       # every INTERVAL_SEC (default 180)
#
# Env:
#   VAST_HOST (default 92.190.14.191 — current vast-affinity; was ssh3.vast.ai)
#   VAST_PORT (default 28225 — current vast-affinity; was 16969)
#   VAST_KEY  (default /tmp/vast_affinity, else Windows path)
#   LOCAL_CHUNKS  ~/s7-affinity-out/chunks
#   REMOTE_OUT    /workspace/s7-out
#   LOW_DISK_KB   2500000   # worker gate in vast_worker_loop.sh (informational)
#   KEEP_REMOTE=1           # mirror only; never delete on Vast
#   INTERVAL_SEC  180
#   LOG           ~/s7-affinity-out/vast_offload.log
#
# Default: every completed chunk is rsynced, byte-verified, then deleted on Vast
# so free space stays above the 2.5G LOW_DISK gate without pausing inference.
set -euo pipefail

HOST="${VAST_HOST:-92.190.14.191}"
PORT="${VAST_PORT:-28225}"
KEY="${VAST_KEY:-}"
LOCAL_CHUNKS="${LOCAL_CHUNKS:-$HOME/s7-affinity-out/chunks}"
REMOTE_OUT="${REMOTE_OUT:-/workspace/s7-out}"
REMOTE_CHUNKS="${REMOTE_OUT}/chunks"
LOW_DISK_KB="${LOW_DISK_KB:-2500000}"
KEEP_REMOTE="${KEEP_REMOTE:-0}"
INTERVAL_SEC="${INTERVAL_SEC:-180}"
LOG="${LOG:-$HOME/s7-affinity-out/vast_offload.log}"

LOOP=0
for arg in "$@"; do
  case "$arg" in
    --loop) LOOP=1 ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
  esac
done

if [[ -z "$KEY" ]]; then
  for candidate in \
    /tmp/vast_affinity \
    "$HOME/.ssh/vast_affinity" \
    /mnt/c/Users/matth/.ssh/vast_affinity
  do
    if [[ -f "$candidate" ]]; then
      KEY="$candidate"
      break
    fi
  done
fi
if [[ -z "${KEY}" || ! -f "$KEY" ]]; then
  echo "ERROR: SSH key not found (set VAST_KEY)" >&2
  exit 2
fi

KEY_USE=/tmp/vast_affinity_offload_key
cp "$KEY" "$KEY_USE"
chmod 600 "$KEY_USE"

SSH=(ssh -p "$PORT" -i "$KEY_USE" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
     -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20)
RSYNC_E="ssh -p ${PORT} -i ${KEY_USE} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

mkdir -p "$LOCAL_CHUNKS" "$(dirname "$LOG")"

log() {
  local msg="[$(date -u +%FT%TZ)] $*"
  echo "$msg" | tee -a "$LOG"
}

free_kb() {
  "${SSH[@]}" "root@$HOST" "df -Pk / | awk 'NR==2{print \$4}'" | tr -d '\r'
}

list_completed() {
  "${SSH[@]}" "root@$HOST" bash -s <<'REMOTE'
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

remote_file_sizes() {
  local cid="$1"
  "${SSH[@]}" "root@$HOST" bash -s <<REMOTE
set -euo pipefail
d=/workspace/s7-out/chunks/${cid}
for f in affinities_core_czyx.npy boundaries_core.tif receipt.json; do
  printf '%s %s\n' "\$f" "\$(stat -c%s "\$d/\$f")"
done
REMOTE
}

verify_local_vs_remote() {
  local cid="$1"
  local local_dir="$LOCAL_CHUNKS/$cid"
  local name remote_sz local_sz
  while read -r name remote_sz; do
    [[ -n "${name:-}" ]] || continue
    remote_sz=$(echo "$remote_sz" | tr -d '\r')
    local_sz=$(stat -c%s "$local_dir/$name")
    if [[ "$local_sz" != "$remote_sz" ]]; then
      echo "SIZE_MISMATCH $cid $name local=$local_sz remote=$remote_sz" >&2
      return 1
    fi
  done < <(remote_file_sizes "$cid")
  return 0
}

delete_remote_chunk() {
  local cid="$1"
  "${SSH[@]}" "root@$HOST" bash -s <<REMOTE
set -euo pipefail
d=/workspace/s7-out/chunks/${cid}
[[ -f "\$d/receipt.json" ]] || { echo "refuse_delete_no_receipt ${cid}"; exit 1; }
s1=\$(stat -c%s "\$d/affinities_core_czyx.npy")
sleep 2
s2=\$(stat -c%s "\$d/affinities_core_czyx.npy")
[[ "\$s1" == "\$s2" ]] || { echo "refuse_delete_growing ${cid} \$s1->\$s2"; exit 1; }
rm -rf "\$d"
echo "deleted ${cid}"
REMOTE
}

offload_once() {
  local avail done_list cid n_mirrored=0 n_del=0 n_skip=0 count=0
  avail=$(free_kb)
  log "free_kb=${avail} low_disk_kb=${LOW_DISK_KB} keep_remote=${KEEP_REMOTE}"

  done_list=$(list_completed | tr -d '\r' || true)
  if [[ -z "${done_list//[[:space:]]/}" ]]; then
    log "no completed chunks on Vast"
    return 0
  fi
  count=$(printf '%s\n' "$done_list" | grep -c . || true)
  log "completed_on_vast=${count}"

  while read -r cid; do
    [[ -n "$cid" ]] || continue
    log "rsync $cid → $LOCAL_CHUNKS/$cid"
    mkdir -p "$LOCAL_CHUNKS/$cid"
    if ! rsync -a --partial -e "$RSYNC_E" \
        "root@$HOST:$REMOTE_CHUNKS/$cid/" "$LOCAL_CHUNKS/$cid/" >>"$LOG" 2>&1; then
      log "ERROR rsync failed $cid — not deleting"
      n_skip=$((n_skip + 1))
      continue
    fi
    if ! verify_local_vs_remote "$cid"; then
      log "ERROR byte verify failed $cid — not deleting"
      n_skip=$((n_skip + 1))
      continue
    fi
    n_mirrored=$((n_mirrored + 1))
    if [[ "$KEEP_REMOTE" == "1" ]]; then
      log "verified $cid — KEEP_REMOTE=1 so left on Vast"
      continue
    fi
    log "verified $cid — deleting remote"
    if delete_remote_chunk "$cid" >>"$LOG" 2>&1; then
      n_del=$((n_del + 1))
    else
      log "ERROR delete refused $cid"
      n_skip=$((n_skip + 1))
    fi
  done <<<"$done_list"

  avail=$(free_kb)
  log "summary mirrored=$n_mirrored deleted=$n_del skipped=$n_skip free_kb=$avail"
  if [[ "$avail" -lt "$LOW_DISK_KB" ]]; then
    log "WARN still below LOW_DISK gate after offload"
  fi
}

if [[ "$LOOP" -eq 1 ]]; then
  log "offload loop start interval=${INTERVAL_SEC}s → $LOCAL_CHUNKS"
  while true; do
    offload_once || log "WARN offload_once rc=$?"
    sleep "$INTERVAL_SEC"
  done
else
  offload_once
fi
