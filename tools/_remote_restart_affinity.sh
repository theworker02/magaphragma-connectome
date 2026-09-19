#!/bin/bash
# Remote restart — args: SHARD WORKER_ID
set -euo pipefail
SHARD="${1:?shard}"
WID="${2:?wid}"
R=/workspace/magaphragma-connectome
test -f "$R/experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json"
python3 -c "import json; d=json.load(open('$R/experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json')); assert d.get('status')=='APPROVED'; print('S6_OK', d['id'])"
test -f "$R/third_party/segneuron/Train_and_Inference/model/Mnet.py" && echo MNET_OK
test -f "$R/tools/s7_infer_accelerate.py" && echo ACCEL_OK
test -f "$R/tools/run_vast_simple.sh" && echo WORKER_OK
pkill -f 'run_affinity_fullvol_s7_fast_worker' 2>/dev/null || true
pkill -f 'run_vast_simple.sh' 2>/dev/null || true
sleep 2
mkdir -p /tmp/s7-logs /tmp/s7-accum /tmp/s7-fast-stage /workspace/s7-out
: > /tmp/s7-logs/worker.log
BATCH=12
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 16000)
if [ -n "${MEM:-}" ] && [ "$MEM" -lt 10000 ]; then BATCH=8; fi
cd "$R"
nohup env S7_SHARD="$SHARD" S7_WORKER_ID="$WID" S7_TILE_BATCH="$BATCH" bash -c 'while true; do bash tools/run_vast_simple.sh >>/tmp/s7-logs/worker.log 2>&1; echo "[$(date -u +%FT%TZ)] exited; respawn" >>/tmp/s7-logs/worker.log; sleep 20; done' >/tmp/s7-logs/loop.log 2>&1 &
echo STARTED_PID=$! BATCH=$BATCH SHARD=$SHARD
sleep 16
echo '--- log ---'
head -n 80 /tmp/s7-logs/worker.log
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
if grep -q "FileNotFoundError.*AFFINITY_S6" /tmp/s7-logs/worker.log; then echo FAIL_S6; exit 3; fi
if grep -q "No module named 'model'" /tmp/s7-logs/worker.log; then echo FAIL_MODEL; exit 4; fi
if grep -qE 'FAST infer|tiles_per_sec|infer_backend|queue pending' /tmp/s7-logs/worker.log; then echo LOOKS_ALIVE; fi
