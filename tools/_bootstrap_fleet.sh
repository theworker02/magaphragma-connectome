#!/usr/bin/env bash
# Bootstrap Affinity lean push across reachable Vast hosts (unique shards).
set -uo pipefail
ROOT="/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome"
LEAN="$ROOT/tools/lean_push_and_start_vast.sh"
export VAST_KEY="${VAST_KEY:-/mnt/c/Users/matth/.ssh/vast_affinity}"

# host:port:shard:worker_id  (3060 Ti already on 0/10)
TARGETS=(
  "ssh2.vast.ai:24958:1/10:vast-2080ti"
  "ssh1.vast.ai:24958:2/10:vast-3060"
  "ssh6.vast.ai:24956:3/10:vast-3080"
  "ssh6.vast.ai:24954:4/10:vast-4060ti-b"
  "ssh5.vast.ai:22704:5/10:vast-4060ti-a"
  "ssh6.vast.ai:22706:6/10:vast-5060ti"
  "ssh3.vast.ai:22708:7/10:vast-a4000-a"
  "ssh6.vast.ai:22712:8/10:vast-a4000-b"
  "14.227.95.149:35774:9/10:vast-5090"
)

ok=0
fail=0
for row in "${TARGETS[@]}"; do
  IFS=':' read -r H P SHARD WID <<<"$row"
  echo "######## BOOTSTRAP $H:$P shard=$SHARD wid=$WID ########"
  if bash "$LEAN" "$H" "$P" "$SHARD" "$WID"; then
    echo "OK $H:$P"
    ok=$((ok+1))
  else
    echo "FAIL $H:$P rc=$?"
    fail=$((fail+1))
  fi
done
echo "bootstrap_done ok=$ok fail=$fail"
