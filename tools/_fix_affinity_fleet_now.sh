#!/usr/bin/env bash
# Emergency fleet repair: S6 + contracts + MNet + restart unique /10 shards.
# Does NOT touch 5090 / virion. Does NOT restart working 3060 Ti unless --retarget-orig.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Prefer Windows path when running under Git Bash
if [[ -d "/c/Users/matth/OneDrive/Desktop/magaphragma-connectome" ]]; then
  ROOT="/c/Users/matth/OneDrive/Desktop/magaphragma-connectome"
fi
KEY="${VAST_KEY:-$HOME/.ssh/vast_affinity}"
[[ -f "$KEY" ]] || KEY="/c/Users/matth/.ssh/vast_affinity"
KU="/tmp/vast_ku_$$"
cp "$KEY" "$KU" && chmod 600 "$KU"
REMOTE="/workspace/magaphragma-connectome"
LOGDIR="/tmp/affinity_fix_$$"
mkdir -p "$LOGDIR"

GATE=(
  "$ROOT/experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json"
  "$ROOT/experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_001.json"
  "$ROOT/experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_002.json"
  "$ROOT/experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_003.json"
)

# host:port:shard:wid:label  — shard 0 reserved for original 3060 Ti
TARGETS=(
  "ssh2.vast.ai:24958:1/10:vast-2080ti:2080Ti"
  "ssh1.vast.ai:24958:2/10:vast-3060:3060"
  "ssh6.vast.ai:24956:3/10:vast-3080:3080"
  "ssh6.vast.ai:24954:4/10:vast-4060ti-b:4060Ti-b"
  "ssh5.vast.ai:22704:5/10:vast-4060ti-a:4060Ti-a"
  "ssh6.vast.ai:22706:6/10:vast-5060ti:5060Ti"
  "ssh3.vast.ai:22708:7/10:vast-a4000-a:A4000-a"
  "ssh6.vast.ai:22712:8/10:vast-a4000-b:A4000-b"
)

fix_one() {
  local H="$1" P="$2" SHARD="$3" WID="$4" LABEL="$5"
  local log="$LOGDIR/${LABEL}.log"
  {
    echo "=== $LABEL $H:$P shard=$SHARD ==="
    local SSH=(ssh -p "$P" -i "$KU" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -o ConnectTimeout=25 -o BatchMode=yes -o ServerAliveInterval=10)
    local SCP=(scp -P "$P" -i "$KU" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -o ConnectTimeout=25 -o BatchMode=yes)
    if ! "${SSH[@]}" "root@$H" "echo SSH_OK; nvidia-smi -L | head -1"; then
      echo "FAIL_SSH"; return 1
    fi
    "${SSH[@]}" "root@$H" "mkdir -p $REMOTE/experiments/phase6e $REMOTE/tools $REMOTE/third_party/segneuron/Train_and_Inference $REMOTE/local_research_build/phase5c-production $REMOTE/experiments/phase6e/AFFINITY-ROI-001/package $REMOTE/experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints /tmp/s7-logs /workspace/s7-out/chunks /tmp/s7-accum /tmp/s7-fast-stage"
    "${SCP[@]}" "${GATE[@]}" "root@$H:$REMOTE/experiments/phase6e/" || { echo FAIL_GATE; return 1; }
    "${SCP[@]}" \
      "$ROOT/tools/run_vast_simple.sh" \
      "$ROOT/tools/run_affinity_fullvol_s7_fast_worker.py" \
      "$ROOT/tools/s7_chunk_claim.py" \
      "$ROOT/tools/s7_durable_commit.py" \
      "$ROOT/tools/s7_infer_accelerate.py" \
      "root@$H:$REMOTE/tools/" || { echo FAIL_TOOLS; return 1; }
    # MNet (critical)
    if command -v rsync >/dev/null 2>&1; then
      rsync -az -e "${SSH[*]}" "$ROOT/third_party/segneuron/Train_and_Inference/" "root@$H:$REMOTE/third_party/segneuron/Train_and_Inference/" || { echo FAIL_MNET; return 1; }
    else
      "${SSH[@]}" "root@$H" "mkdir -p $REMOTE/third_party/segneuron/Train_and_Inference/model"
      "${SCP[@]}" -r "$ROOT/third_party/segneuron/Train_and_Inference/model" "root@$H:$REMOTE/third_party/segneuron/Train_and_Inference/" || { echo FAIL_MNET; return 1; }
    fi
    # chunks / ROI / ckpt if missing
    "${SSH[@]}" "root@$H" "test -f $REMOTE/local_research_build/phase5c-production/chunks.json" \
      || "${SCP[@]}" "$ROOT/local_research_build/phase5c-production/chunks.json" "root@$H:$REMOTE/local_research_build/phase5c-production/"
    if ! "${SSH[@]}" "root@$H" "test -f $REMOTE/experiments/phase6e/AFFINITY-ROI-001/package/ROI_PACKAGE.json"; then
      if command -v rsync >/dev/null 2>&1; then
        rsync -az -e "${SSH[*]}" "$ROOT/experiments/phase6e/AFFINITY-ROI-001/package/" "root@$H:$REMOTE/experiments/phase6e/AFFINITY-ROI-001/package/"
      else
        "${SCP[@]}" -r "$ROOT/experiments/phase6e/AFFINITY-ROI-001/package/." "root@$H:$REMOTE/experiments/phase6e/AFFINITY-ROI-001/package/"
      fi
    fi
    local CKPT_DST="$REMOTE/experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints"
    "${SSH[@]}" "root@$H" "test -f $CKPT_DST/checkpoint-step10.pt" \
      || "${SCP[@]}" "$ROOT/experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt" "root@$H:$CKPT_DST/"

    "${SSH[@]}" "root@$H" bash -s <<REMOTE_SCRIPT
set -euo pipefail
test -f $REMOTE/experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json
python3 -c "import json; assert json.load(open('$REMOTE/experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json'))['status']=='APPROVED'; print('S6_OK')"
test -f $REMOTE/third_party/segneuron/Train_and_Inference/model/Mnet.py && echo MNET_OK
test -f $REMOTE/tools/s7_infer_accelerate.py && echo ACCEL_OK
pkill -f 'run_affinity_fullvol_s7_fast_worker' 2>/dev/null || true
pkill -f 'run_vast_simple.sh' 2>/dev/null || true
sleep 2
mkdir -p /tmp/s7-logs /tmp/s7-accum /tmp/s7-fast-stage
: > /tmp/s7-logs/worker.log
BATCH=12
MEM=\$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 16000)
if [[ -n "\${MEM:-}" && "\$MEM" -lt 10000 ]]; then BATCH=8; fi
cd $REMOTE
nohup env S7_SHARD=$SHARD S7_WORKER_ID=$WID S7_TILE_BATCH=\$BATCH bash -c 'while true; do
  bash tools/run_vast_simple.sh >>/tmp/s7-logs/worker.log 2>&1
  echo "[\$(date -u +%FT%TZ)] exited; respawn 20s" >>/tmp/s7-logs/worker.log
  sleep 20
done' >/tmp/s7-logs/loop.log 2>&1 &
echo STARTED BATCH=\$BATCH SHARD=$SHARD
sleep 14
echo '--- log ---'
head -n 80 /tmp/s7-logs/worker.log
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
if grep -q "FileNotFoundError.*AFFINITY_S6" /tmp/s7-logs/worker.log; then echo FAIL_S6; exit 3; fi
if grep -q "No module named 'model'" /tmp/s7-logs/worker.log; then echo FAIL_MODEL; exit 4; fi
if grep -qE 'FAST infer|tiles_per_sec|infer_backend|queue pending' /tmp/s7-logs/worker.log; then echo LOOKS_ALIVE; fi
REMOTE_SCRIPT
  } >"$log" 2>&1
  local rc=$?
  echo "$([ $rc -eq 0 ] && echo OK || echo FAIL) $LABEL rc=$rc"
  return $rc
}

confirm_orig() {
  {
    echo "=== 3060Ti-orig confirm (do not restart unless broken) ==="
    ssh -p 28225 -i "$KU" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes \
      -o ConnectTimeout=20 -o BatchMode=yes root@92.190.14.191 \
      'pgrep -af run_affinity_fullvol | head -3; tr "\0" "\n" < /proc/$(pgrep -f run_affinity_fullvol_s7_fast_worker | head -1)/environ 2>/dev/null | grep S7_ || true; tail -8 /tmp/s7-logs/worker.log 2>/dev/null || tail -8 /workspace/s7-out/worker.log 2>/dev/null; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader'
  } >"$LOGDIR/3060Ti-orig.log" 2>&1
  echo "ORIG_DONE"
}

# Optional: retarget orig to 0/10 WITHOUT killing if already on 0/10 and inferring
retarget_orig_if_needed() {
  {
    echo "=== retarget orig check ==="
    ssh -p 28225 -i "$KU" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes \
      -o ConnectTimeout=20 -o BatchMode=yes root@92.190.14.191 bash -s <<'R'
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
SH=$(tr '\0' '\n' < /proc/$(pgrep -f run_affinity_fullvol_s7_fast_worker | head -1)/environ 2>/dev/null | grep '^S7_SHARD=' || true)
echo "util=$UTIL shard_env=$SH"
# If util>0 and shard is 0/10, leave alone
if [[ "${UTIL:-0}" -gt 5 ]] && echo "$SH" | grep -q '0/10'; then
  echo LEAVE_ORIG_ALONE
  exit 0
fi
# If inferring on 1/2, retarget to 0/10
if echo "$SH" | grep -qE '1/2|0/10' || [[ "${UTIL:-0}" -eq 0 ]]; then
  echo RETARGET_TO_0_10
  pkill -f run_affinity_fullvol_s7_fast_worker 2>/dev/null || true
  pkill -f run_vast_simple.sh 2>/dev/null || true
  sleep 2
  # Ensure S6 present (should already be)
  test -f /workspace/magaphragma-connectome/experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json || exit 5
  cd /workspace/magaphragma-connectome
  nohup env S7_SHARD=0/10 S7_WORKER_ID=vast-3060ti S7_TILE_BATCH=8 bash -c 'while true; do bash tools/run_vast_simple.sh >>/tmp/s7-logs/worker.log 2>&1; sleep 20; done' >/tmp/s7-logs/loop.log 2>&1 &
  sleep 8
  tail -15 /tmp/s7-logs/worker.log
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
fi
R
  } >"$LOGDIR/3060Ti-retarget.log" 2>&1
  echo "RETARGET_DONE"
}

pids=()
for row in "${TARGETS[@]}"; do
  IFS=':' read -r H P SHARD WID LABEL <<<"$row"
  fix_one "$H" "$P" "$SHARD" "$WID" "$LABEL" &
  pids+=($!)
done
confirm_orig &
pids+=($!)
retarget_orig_if_needed &
pids+=($!)

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done

echo "==== SUMMARIES ===="
for f in "$LOGDIR"/*.log; do
  echo "----- $(basename "$f") -----"
  tail -n 35 "$f" || true
  echo
done
echo "fix_done fail=$fail logs=$LOGDIR"
