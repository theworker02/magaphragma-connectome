#!/usr/bin/env bash
# Local side: keep draining shard 0/2 on RX 7800 XT.
# Tuned: batch 16 + torch.compile (~1.1x vs eager on FAST_003 bench).
set -euo pipefail
cd /mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
PY=/opt/venvs/unlearning-rocm/bin/python
export S7_WORKER_ID=local-rx7800xt-01
export S7_CLAIM_BACKEND=local
export S7_DURABLE_MODE=local
export S7_TILE_BATCH=16
export S7_INFER_BACKEND=compile
# OneDrive /mnt/c OUT breaks atomic stage→final moves (EPERM). Keep durable tree on Linux FS.
export S7_OUT="${S7_OUT:-$HOME/s7-affinity-out}"
export S7_STAGE_DIR="${S7_STAGE_DIR:-/tmp/s7-fast-stage}"
export S7_DIRECT_SAVE="${S7_DIRECT_SAVE:-1}"
mkdir -p "$S7_OUT" "$S7_STAGE_DIR"

# Stop old worker if you want a clean shard restart (optional)
# pkill -f run_affinity_fullvol_s7_fast_worker.py || true

exec "$PY" -u tools/run_affinity_fullvol_s7_fast_worker.py \
  --claim --loop --max-chunks 64 --batch-size 16 \
  --contract FAST_003 --infer-backend compile \
  --shard 0/2
