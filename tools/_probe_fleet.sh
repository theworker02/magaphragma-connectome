#!/usr/bin/env bash
# Probe Affinity Vast fleet SSH + worker state.
set -u
KEY="${VAST_KEY:-/mnt/c/Users/matth/.ssh/vast_affinity}"
KEY_USE=/tmp/vast_k_probe_$$
cp "$KEY" "$KEY_USE" && chmod 600 "$KEY_USE"
hosts=(
  "92.190.14.191:28225"
  "14.227.95.149:35774"
  "122.51.254.66:22704"
  "74.96.198.230:22706"
  "202.122.49.242:22708"
  "38.29.145.27:22712"
  "122.51.254.66:24954"
  "122.51.254.66:24956"
  "58.8.189.113:24958"
  "99.127.80.211:24958"
)
for hp in "${hosts[@]}"; do
  H=${hp%%:*}
  P=${hp##*:}
  echo "=== $H:$P ==="
  ssh -p "$P" -i "$KEY_USE" \
    -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes \
    -o ConnectTimeout=15 -o BatchMode=yes \
    "root@$H" bash -s <<'REMOTE' 2>&1 | head -30
hostname || true
nvidia-smi -L 2>/dev/null | head -1 || echo NO_GPU
pgrep -af 'run_affinity_fullvol|run_vast_simple' 2>/dev/null | grep -v grep | head -5 || echo NO_WORKER
if [[ -f /tmp/s7-logs/worker.log ]]; then
  echo "--- worker.log tail ---"
  tail -n 8 /tmp/s7-logs/worker.log
elif [[ -f /workspace/s7-out/worker.log ]]; then
  echo "--- s7-out worker.log tail ---"
  tail -n 8 /workspace/s7-out/worker.log
fi
test -f /workspace/magaphragma-connectome/tools/run_vast_simple.sh && echo PUSH_OK || echo NO_PUSH
df -h /workspace 2>/dev/null | tail -1 || true
REMOTE
  echo
done
rm -f "$KEY_USE"
