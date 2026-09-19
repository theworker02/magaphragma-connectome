#!/usr/bin/env bash
# Restart local shard 0/2 with Linux-native S7_OUT (avoid OneDrive EPERM) + direct save.
set -euo pipefail
REPO=/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
PY=/opt/venvs/unlearning-rocm/bin/python
OUT_NEW=$HOME/s7-affinity-out
OLD_OUT=$REPO/experiments/phase6e/AFFINITY-FULLVOL-S7-001
LOG=$OUT_NEW/local_worker.log

mkdir -p "$OUT_NEW/chunks" "$OUT_NEW/claims"
# Carry forward queue progress so we don't redo completed chunks.
if [[ -f $OLD_OUT/queue_state.json && ! -f $OUT_NEW/queue_state.json ]]; then
  cp "$OLD_OUT/queue_state.json" "$OUT_NEW/queue_state.json"
  echo "copied queue_state from OneDrive out"
fi

# Stop old workers (including root-owned)
pkill -f 'run_affinity_fullvol_s7_fast_worker.py' 2>/dev/null || true
sleep 1
if pgrep -f 'run_affinity_fullvol_s7_fast_worker.py' >/dev/null; then
  # may need root
  true
fi

export S7_WORKER_ID=local-rx7800xt-01
export S7_CLAIM_BACKEND=local
export S7_DURABLE_MODE=local
export S7_TILE_BATCH=16
export S7_INFER_BACKEND=compile
export S7_OUT=$OUT_NEW
export S7_STAGE_DIR=/tmp/s7-fast-stage
export S7_DIRECT_SAVE=1
mkdir -p "$S7_STAGE_DIR"
chmod 777 "$S7_STAGE_DIR" 2>/dev/null || true

cd "$REPO"
nohup "$PY" -u tools/run_affinity_fullvol_s7_fast_worker.py \
  --claim --loop --max-chunks 64 --batch-size 16 \
  --contract FAST_003 --infer-backend compile \
  --shard 0/2 > "$LOG" 2>&1 &
echo "pid=$! out=$OUT_NEW"
sleep 12
pgrep -af run_affinity_fullvol_s7_fast_worker || echo none
tail -n 30 "$LOG"
