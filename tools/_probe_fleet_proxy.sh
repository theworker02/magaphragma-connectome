#!/usr/bin/env bash
# Probe Affinity fleet via Vast SSH proxy endpoints.
set -u
KEY="${VAST_KEY:-/mnt/c/Users/matth/.ssh/vast_affinity}"
KEY_USE=/tmp/vast_k_probe2_$$
cp "$KEY" "$KEY_USE" && chmod 600 "$KEY_USE"
# id|proxy_host:port|note
hosts=(
  "51496969|ssh3.vast.ai:16968|3060Ti-keep"
  "51510290|ssh2.vast.ai:30290|5090"
  "51622705|ssh5.vast.ai:22704|4060Ti"
  "51622707|ssh6.vast.ai:22706|5060Ti"
  "51622708|ssh3.vast.ai:22708|A4000"
  "51622713|ssh6.vast.ai:22712|A4000"
  "51624954|ssh6.vast.ai:24954|4060Ti-new"
  "51624956|ssh6.vast.ai:24956|3080-new"
  "51624958|ssh2.vast.ai:24958|2080Ti-new"
  "51624959|ssh1.vast.ai:24958|3060-new"
)
for row in "${hosts[@]}"; do
  IFS='|' read -r ID HP NOTE <<<"$row"
  H=${HP%%:*}
  P=${HP##*:}
  echo "=== id=$ID $H:$P ($NOTE) ==="
  ssh -p "$P" -i "$KEY_USE" \
    -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes \
    -o ConnectTimeout=20 -o BatchMode=yes \
    "root@$H" bash -s <<'REMOTE' 2>&1 | head -25
hostname || true
nvidia-smi -L 2>/dev/null | head -1 || echo NO_GPU
pgrep -af 'run_affinity_fullvol|run_vast_simple' 2>/dev/null | grep -v grep | head -3 || echo NO_WORKER
test -f /workspace/magaphragma-connectome/tools/run_vast_simple.sh && echo PUSH_OK || echo NO_PUSH
df -h / /workspace /tmp 2>/dev/null | sed -n '1,5p' || true
REMOTE
  echo
done
rm -f "$KEY_USE"
