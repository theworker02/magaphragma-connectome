#!/usr/bin/env bash
# Verbose SSH connectivity matrix (direct + proxy).
set -u
KEY="${VAST_KEY:-/mnt/c/Users/matth/.ssh/vast_affinity}"
K=/tmp/vast_mat_$$
cp "$KEY" "$K" && chmod 600 "$K"
try() {
  local label="$1" host="$2" port="$3"
  echo -n "$label -> "
  out=$(ssh -p "$port" -i "$K" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes \
    -o ConnectTimeout=12 -o BatchMode=yes -o PreferredAuthentications=publickey \
    "root@$host" "echo OK; nvidia-smi -L | head -1" 2>&1) || true
  if echo "$out" | grep -q '^OK'; then
    echo "OK | $(echo "$out" | grep -v '^OK' | head -1)"
  else
    echo "FAIL | $(echo "$out" | tr '\n' ' ' | tail -c 160)"
  fi
}
# From fleet summary: direct IP:mapped_ssh and proxy
try "3060ti-direct" 92.190.14.191 28225
try "3060ti-proxy" ssh3.vast.ai 16968
try "5090-direct" 14.227.95.149 35774
try "5090-proxy" ssh2.vast.ai 30290
try "4060a-proxy" ssh5.vast.ai 22704
try "5060-proxy" ssh6.vast.ai 22706
try "a4000a-proxy" ssh3.vast.ai 22708
try "a4000b-proxy" ssh6.vast.ai 22712
try "4060b-proxy" ssh6.vast.ai 24954
try "3080-proxy" ssh6.vast.ai 24956
try "2080-proxy" ssh2.vast.ai 24958
try "3060n-proxy" ssh1.vast.ai 24958
try "3080-direct" 122.51.254.66 24956
try "4060b-direct" 122.51.254.66 24954
rm -f "$K"
