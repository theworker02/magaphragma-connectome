#!/usr/bin/env bash
# Lean push Affinity essentials to one Vast host and start shard worker.
# Usage: bash tools/lean_push_and_start_vast.sh HOST PORT S7_SHARD [WORKER_ID]
set -euo pipefail
HOST="${1:?host}"
PORT="${2:?port}"
SHARD="${3:?shard e.g. 2/6}"
WORKER_ID="${4:-vast-$(echo "$SHARD" | tr '/' '-')}"
KEY="${VAST_KEY:-/mnt/c/Users/matth/.ssh/vast_affinity}"
ROOT="/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome"
REMOTE="/workspace/magaphragma-connectome"
CKPT_SRC="$ROOT/experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt"
CKPT_DST="$REMOTE/experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints"

KEY_USE=/tmp/vast_affinity_push_key_$$
cp "$KEY" "$KEY_USE" && chmod 600 "$KEY_USE"
SSH=(ssh -p "$PORT" -i "$KEY_USE" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -o ConnectTimeout=30 -o BatchMode=yes)
SCP=(scp -P "$PORT" -i "$KEY_USE" -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -o ConnectTimeout=30 -o BatchMode=yes)

echo "=== lean push $HOST:$PORT shard=$SHARD worker=$WORKER_ID ==="
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  if "${SSH[@]}" "root@$HOST" "echo SSH_OK && nvidia-smi -L"; then
    break
  fi
  echo "waiting for SSH/GPU... ($i)"
  sleep 15
done

"${SSH[@]}" "root@$HOST" "mkdir -p $REMOTE/tools $REMOTE/tools/cloud $CKPT_DST $REMOTE/experiments/phase6e/AFFINITY-ROI-001/package $REMOTE/local_research_build/phase5c-production $REMOTE/third_party/segneuron/Train_and_Inference /workspace/s7-out/chunks /tmp/s7-accum /tmp/s7-fast-stage"

"${SCP[@]}" \
  "$ROOT/tools/run_vast_simple.sh" \
  "$ROOT/tools/run_affinity_fullvol_s7_fast_worker.py" \
  "$ROOT/tools/s7_chunk_claim.py" \
  "$ROOT/tools/s7_durable_commit.py" \
  "$ROOT/tools/s7_infer_accelerate.py" \
  "root@$HOST:$REMOTE/tools/"

# Optional helpers if present
for f in s7_volume_io.py qualify_nvidia_affinity.py; do
  [[ -f "$ROOT/tools/$f" ]] && "${SCP[@]}" "$ROOT/tools/$f" "root@$HOST:$REMOTE/tools/" || true
done

"${SCP[@]}" "$CKPT_SRC" "root@$HOST:$CKPT_DST/"
"${SCP[@]}" "$ROOT/local_research_build/phase5c-production/chunks.json" "root@$HOST:$REMOTE/local_research_build/phase5c-production/" || true

# Worker gate / contracts — missing S6 causes FileNotFoundError crash-loops every respawn
"${SSH[@]}" "root@$HOST" "mkdir -p $REMOTE/experiments/phase6e $REMOTE/third_party/segneuron/Train_and_Inference"
"${SCP[@]}" \
  "$ROOT/experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json" \
  "$ROOT/experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_001.json" \
  "$ROOT/experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_002.json" \
  "$ROOT/experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_003.json" \
  "root@$HOST:$REMOTE/experiments/phase6e/"

# MNet import path (third_party/segneuron/Train_and_Inference/model)
rsync -az -e "${SSH[*]}" \
  "$ROOT/third_party/segneuron/Train_and_Inference/" \
  "root@$HOST:$REMOTE/third_party/segneuron/Train_and_Inference/"

# ROI / funnel metadata (small)
if [[ -d "$ROOT/experiments/phase6e/AFFINITY-ROI-001/package" ]]; then
  rsync -az -e "${SSH[*]}" "$ROOT/experiments/phase6e/AFFINITY-ROI-001/package/" "root@$HOST:$REMOTE/experiments/phase6e/AFFINITY-ROI-001/package/" || true
fi

"${SSH[@]}" "root@$HOST" bash -s <<REMOTE
set -euo pipefail
chmod +x $REMOTE/tools/*.sh || true
test -f $CKPT_DST/checkpoint-step10.pt && echo CKPT_OK || { echo CKPT_MISSING; exit 2; }
# Disk hygiene: keep offload from filling disk — clear caches, keep memmap on /tmp
rm -rf /root/.cache/pip /workspace/.cache 2>/dev/null || true
df -h / /tmp /workspace | sed -n '1,4p'
# Restart worker on this shard
pkill -f 'run_affinity_fullvol_s7_fast_worker' 2>/dev/null || true
pkill -f 'run_vast_simple.sh' 2>/dev/null || true
pkill -f 'vast_worker_loop' 2>/dev/null || true
sleep 2
mkdir -p /workspace/s7-out /tmp/s7-accum /tmp/s7-fast-stage /tmp/s7-logs
export S7_SHARD='$SHARD'
export S7_WORKER_ID='$WORKER_ID'
export S7_TILE_BATCH="\${S7_TILE_BATCH:-12}"
export S7_ACCUM_DTYPE=float16
export S7_ACCUM_MEMMAP_DIR=/tmp/s7-accum
export S7_STREAM_SAVE=1
cd $REMOTE
nohup bash -c 'while true; do
  S7_SHARD=$SHARD S7_WORKER_ID=$WORKER_ID bash tools/run_vast_simple.sh >>/tmp/s7-logs/worker.log 2>&1
  echo "[\$(date -u +%FT%TZ)] worker exited; respawn in 20s" >>/tmp/s7-logs/worker.log
  sleep 20
done' >/tmp/s7-logs/loop.log 2>&1 &
echo STARTED_PID=\$!
sleep 3
tail -n 30 /tmp/s7-logs/worker.log || true
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader || true
REMOTE

rm -f "$KEY_USE"
echo "DONE $HOST:$PORT $SHARD"
