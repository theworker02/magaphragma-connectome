#!/usr/bin/env bash
# Push missing gate JSON files + restart workers; retry SSH-failed hosts.
set -uo pipefail
ROOT="/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome"
KEY="${VAST_KEY:-/mnt/c/Users/matth/.ssh/vast_affinity}"
K=/tmp/vast_fix_$$
cp "$KEY" "$K" && chmod 600 "$K"
REMOTE=/workspace/magaphragma-connectome
PHASE6=$REMOTE/experiments/phase6e

# Files required by worker gate checks
FILES=(
  "experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json"
  "experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_001.json"
  "experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_002.json"
  "experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_003.json"
)

# Working hosts already bootstrapped (need gate files)
FIX_HOSTS=(
  "92.190.14.191:28225:0/10:vast-3060ti"
  "ssh2.vast.ai:24958:1/10:vast-2080ti"
  "ssh1.vast.ai:24958:2/10:vast-3060"
  "ssh6.vast.ai:24956:3/10:vast-3080"
  "ssh6.vast.ai:24954:4/10:vast-4060ti-b"
  "ssh5.vast.ai:22704:5/10:vast-4060ti-a"
  "ssh6.vast.ai:22712:8/10:vast-a4000-b"
  "14.227.95.149:35774:9/10:vast-5090"
)

# Retry previously failed
RETRY_HOSTS=(
  "ssh6.vast.ai:22706:6/10:vast-5060ti"
  "ssh3.vast.ai:22708:7/10:vast-a4000-a"
)

fix_one() {
  local H=$1 P=$2 SHARD=$3 WID=$4
  echo "=== fix $H:$P $SHARD ==="
  local SSH=(ssh -p "$P" -i "$K" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -o ConnectTimeout=20 -o BatchMode=yes)
  local SCP=(scp -P "$P" -i "$K" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -o ConnectTimeout=20 -o BatchMode=yes)
  if ! "${SSH[@]}" "root@$H" "mkdir -p $PHASE6 && echo SSH_OK"; then
    echo "SSH_FAIL $H:$P"
    return 1
  fi
  for f in "${FILES[@]}"; do
    if [[ -f "$ROOT/$f" ]]; then
      "${SCP[@]}" "$ROOT/$f" "root@$H:$REMOTE/$f" || return 1
    else
      echo "MISSING_LOCAL $f"
    fi
  done
  "${SSH[@]}" "root@$H" bash -s <<REMOTE
set -euo pipefail
pkill -f 'run_affinity_fullvol_s7_fast_worker' 2>/dev/null || true
pkill -f 'run_vast_simple.sh' 2>/dev/null || true
sleep 2
mkdir -p /tmp/s7-accum /tmp/s7-fast-stage /tmp/s7-logs /workspace/s7-out
test -f $PHASE6/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json && echo S6_OK || { echo S6_MISSING; exit 2; }
cd $REMOTE
nohup bash -c 'while true; do
  S7_SHARD=$SHARD S7_WORKER_ID=$WID S7_ACCUM_DTYPE=float16 S7_ACCUM_MEMMAP_DIR=/tmp/s7-accum S7_STREAM_SAVE=1 \
    bash tools/run_vast_simple.sh >>/tmp/s7-logs/worker.log 2>&1
  echo "[\$(date -u +%FT%TZ)] exited; respawn" >>/tmp/s7-logs/worker.log
  sleep 20
done' >/tmp/s7-logs/loop.log 2>&1 &
sleep 5
tail -n 25 /tmp/s7-logs/worker.log || true
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader || true
REMOTE
}

for row in "${FIX_HOSTS[@]}" "${RETRY_HOSTS[@]}"; do
  IFS=':' read -r H P SHARD WID <<<"$row"
  fix_one "$H" "$P" "$SHARD" "$WID" || echo "FAIL $WID"
done

# Full lean push for retries that never got code
LEAN="$ROOT/tools/lean_push_and_start_vast.sh"
export VAST_KEY="$KEY"
for row in "${RETRY_HOSTS[@]}"; do
  IFS=':' read -r H P SHARD WID <<<"$row"
  echo "=== lean retry $WID ==="
  bash "$LEAN" "$H" "$P" "$SHARD" "$WID" || echo "LEAN_FAIL $WID"
  # also push gate files after lean
  fix_one "$H" "$P" "$SHARD" "$WID" || true
done

rm -f "$K"
echo ALL_FIX_DONE
