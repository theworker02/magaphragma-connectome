#!/usr/bin/env bash
# Parallel lean-push bootstrap for remaining Affinity hosts.
set -uo pipefail
ROOT="/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome"
LEAN="$ROOT/tools/lean_push_and_start_vast.sh"
export VAST_KEY="${VAST_KEY:-/mnt/c/Users/matth/.ssh/vast_affinity}"
LOGDIR="/tmp/affinity_bootstrap_$$"
mkdir -p "$LOGDIR"

TARGETS=(
  "ssh1.vast.ai:24958:2/10:vast-3060"
  "ssh6.vast.ai:24956:3/10:vast-3080"
  "ssh6.vast.ai:24954:4/10:vast-4060ti-b"
  "ssh5.vast.ai:22704:5/10:vast-4060ti-a"
  "ssh6.vast.ai:22706:6/10:vast-5060ti"
  "ssh3.vast.ai:22708:7/10:vast-a4000-a"
  "ssh6.vast.ai:22712:8/10:vast-a4000-b"
  "14.227.95.149:35774:9/10:vast-5090"
)

pids=()
for row in "${TARGETS[@]}"; do
  IFS=':' read -r H P SHARD WID <<<"$row"
  log="$LOGDIR/${WID}.log"
  echo "launch $H:$P $SHARD -> $log"
  (
    if bash "$LEAN" "$H" "$P" "$SHARD" "$WID" >"$log" 2>&1; then
      echo "OK $WID" | tee -a "$log"
    else
      echo "FAIL $WID rc=$?" | tee -a "$log"
    fi
  ) &
  pids+=($!)
done

echo "waiting ${#pids[@]} jobs..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=$((fail+1)); fi
done

echo "==== summaries ===="
for row in "${TARGETS[@]}"; do
  IFS=':' read -r H P SHARD WID <<<"$row"
  echo "--- $WID ---"
  tail -n 15 "$LOGDIR/${WID}.log" || true
done
echo "parallel_bootstrap_done fail=$fail logs=$LOGDIR"
