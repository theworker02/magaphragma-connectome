#!/usr/bin/env bash
# Emergency: push S6 auth + FAST contracts to all Affinity hosts, restart unique /10 shards.
# Does NOT touch 5090 until we inspect for virion CPT.
set -uo pipefail
ROOT="/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome"
KEY="${VAST_KEY:-/mnt/c/Users/matth/.ssh/vast_affinity}"
KEY_USE=/tmp/vast_s6_gate_$$
cp "$KEY" "$KEY_USE" && chmod 600 "$KEY_USE"
REMOTE="/workspace/magaphragma-connectome"
PHASE6E="$REMOTE/experiments/phase6e"
LOGDIR="/tmp/affinity_s6_push_$$"
mkdir -p "$LOGDIR"

GATE_FILES=(
  "$ROOT/experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json"
  "$ROOT/experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_001.json"
  "$ROOT/experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_002.json"
  "$ROOT/experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_003.json"
)

# host:port:shard:worker_id:label
TARGETS=(
  "92.190.14.191:28225:0/10:vast-3060ti:3060Ti-orig"
  "ssh2.vast.ai:24958:1/10:vast-2080ti:2080Ti"
  "ssh1.vast.ai:24958:2/10:vast-3060:3060"
  "ssh6.vast.ai:24956:3/10:vast-3080:3080"
  "ssh6.vast.ai:24954:4/10:vast-4060ti-b:4060Ti-b"
  "ssh5.vast.ai:22704:5/10:vast-4060ti-a:4060Ti-a"
  "ssh6.vast.ai:22706:6/10:vast-5060ti:5060Ti"
  "ssh3.vast.ai:22708:7/10:vast-a4000-a:A4000-a"
  "ssh6.vast.ai:22712:8/10:vast-a4000-b:A4000-b"
)

push_one() {
  local H="$1" P="$2" SHARD="$3" WID="$4" LABEL="$5"
  local log="$LOGDIR/${LABEL}.log"
  {
    echo "=== $LABEL $H:$P shard=$SHARD ==="
    local SSH=(ssh -p "$P" -i "$KEY_USE" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -o ConnectTimeout=25 -o BatchMode=yes -o ServerAliveInterval=10)
    local SCP=(scp -P "$P" -i "$KEY_USE" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -o ConnectTimeout=25 -o BatchMode=yes)
    if ! "${SSH[@]}" "root@$H" "echo SSH_OK && nvidia-smi -L | head -1"; then
      echo "FAIL_SSH $LABEL"
      return 1
    fi
    "${SSH[@]}" "root@$H" "mkdir -p $PHASE6E $REMOTE/tools /tmp/s7-logs /workspace/s7-out/chunks /tmp/s7-accum /tmp/s7-fast-stage"
    "${SCP[@]}" "${GATE_FILES[@]}" "root@$H:$PHASE6E/" || { echo "FAIL_SCP_GATE $LABEL"; return 1; }
    # Ensure worker scripts present (may already be there)
    "${SCP[@]}" \
      "$ROOT/tools/run_vast_simple.sh" \
      "$ROOT/tools/run_affinity_fullvol_s7_fast_worker.py" \
      "$ROOT/tools/s7_chunk_claim.py" \
      "$ROOT/tools/s7_durable_commit.py" \
      "root@$H:$REMOTE/tools/" || true
    # ROI package if missing (worker needs ROI_PACKAGE.json)
    "${SSH[@]}" "root@$H" "test -f $PHASE6E/AFFINITY-ROI-001/package/ROI_PACKAGE.json" \
      || rsync -az -e "${SSH[*]}" "$ROOT/experiments/phase6e/AFFINITY-ROI-001/package/" "root@$H:$PHASE6E/AFFINITY-ROI-001/package/" || true
    # Verify gates on remote
    "${SSH[@]}" "root@$H" bash -s <<REMOTE
set -euo pipefail
test -f $PHASE6E/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json
python3 -c "import json; d=json.load(open('$PHASE6E/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json')); assert d.get('status')=='APPROVED'; print('S6_OK', d['id'])"
test -f $PHASE6E/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_003.json && echo FAST003_OK
# Kill prior Affinity workers only (not unrelated jobs)
pkill -f 'run_affinity_fullvol_s7_fast_worker' 2>/dev/null || true
pkill -f 'run_vast_simple.sh' 2>/dev/null || true
pkill -f 'vast_worker_loop' 2>/dev/null || true
sleep 2
# Clear stale crash loop noise marke
mkdir -p /tmp/s7-logs /tmp/s7-accum /tmp/s7-fast-stage
# Default batch; 8GB cards use 8
BATCH=12
MEM=\$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [[ -n "\${MEM:-}" && "\$MEM" -lt 10000 ]]; then BATCH=8; fi
cd $REMOTE
nohup bash -c "while true; do
  S7_SHARD=$SHARD S7_WORKER_ID=$WID S7_TILE_BATCH=\$BATCH bash tools/run_vast_simple.sh >>/tmp/s7-logs/worker.log 2>&1
  echo \"[\$(date -u +%FT%TZ)] worker exited; respawn in 20s\" >>/tmp/s7-logs/worker.log
  sleep 20
done" >/tmp/s7-logs/loop.log 2>&1 &
echo STARTED_PID=\$! BATCH=\$BATCH SHARD=$SHARD
sleep 8
echo '--- log ---'
tail -n 40 /tmp/s7-logs/worker.log || true
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader || true
# Detect immediate S6 miss
if grep -q 'AFFINITY_S6_SEG_QC_AUTHORIZATION' /tmp/s7-logs/worker.log 2>/dev/null && grep -qi 'No such file\|FileNotFoundError' /tmp/s7-logs/worker.log 2>/dev/null; then
  echo STILL_MISSING_S6
  exit 3
fi
if grep -qi 'FAST infer\|CUDA\|queue pending\|claim' /tmp/s7-logs/worker.log 2>/dev/null; then
  echo LOOKS_ALIVE
fi
REMOTE
  } >"$log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then echo "OK $LABEL"; else echo "FAIL $LABEL rc=$rc"; fi
  return $rc
}

# Inspect 5090 separately — do not kill virion blindly
inspect_5090() {
  local H=14.227.95.149 P=35774
  local log="$LOGDIR/5090-inspect.log"
  {
    echo "=== 5090 inspect $H:$P ==="
    ssh -p "$P" -i "$KEY_USE" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes \
      -o ConnectTimeout=25 -o BatchMode=yes "root@$H" bash -s <<'REMOTE' || echo FAIL_SSH_5090
hostname || true
nvidia-smi -L 2>/dev/null | head -2 || true
echo '--- processes ---'
ps aux 2>/dev/null | grep -E 'python|train|virion|affinity|cpt|run_' | grep -v grep | head -40 || true
echo '--- nvidia ---'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || nvidia-smi | head -40
echo '--- labels/env hints ---'
cat /etc/vast_containerlabel 2>/dev/null || true
env | grep -iE 'VAST|AFFINITY|LABEL|ONSTART' 2>/dev/null | head -20 || true
ls /workspace 2>/dev/null | head -40 || true
pgrep -af 'virion|affinity|cpt' 2>/dev/null | head -20 || true
REMOTE
  } >"$log" 2>&1
  echo "5090_INSPECT_DONE log=$log"
}

pids=()
for row in "${TARGETS[@]}"; do
  IFS=':' read -r H P SHARD WID LABEL <<<"$row"
  push_one "$H" "$P" "$SHARD" "$WID" "$LABEL" &
  pids+=($!)
done
inspect_5090 &
pids+=($!)

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=$((fail+1)); fi
done

echo "==== summaries ===="
for row in "${TARGETS[@]}"; do
  IFS=':' read -r H P SHARD WID LABEL <<<"$row"
  echo "----- $LABEL ($SHARD) -----"
  tail -n 25 "$LOGDIR/${LABEL}.log" || true
  echo
done
echo "----- 5090 -----"
tail -n 40 "$LOGDIR/5090-inspect.log" || true
echo "push_s6_done fail=$fail logs=$LOGDIR"
