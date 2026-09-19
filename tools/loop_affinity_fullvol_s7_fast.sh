#!/usr/bin/env bash
set -euo pipefail
PY=/opt/venvs/unlearning-rocm/bin/python
ROOT=/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
WORKER=$ROOT/tools/run_affinity_fullvol_s7_fast_worker.py
export S7_CLAIM_BACKEND="${S7_CLAIM_BACKEND:-local}"
# compile keeps dying (exit 1 / SIGKILL); default to eager until stable
export S7_INFER_BACKEND="${S7_INFER_BACKEND:-eager}"
export S7_TILE_BATCH="${S7_TILE_BATCH:-16}"
if [[ "${S7_INFER_BACKEND}" == "migraphx" ]]; then
  export ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.2.1}"
  export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/migraphx/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${ROCM_PATH}/lib:${PYTHONPATH:-}"
fi
LOG="${S7_WORKER_LOG:-$ROOT/experiments/phase6e/AFFINITY-FULLVOL-S7-001/worker_fast003.log}"
mkdir -p "$(dirname "$LOG")"
echo starting_continuous_loop_FAST_003_claim "${S7_INFER_BACKEND}" | tee -a "$LOG"
exec "$PY" -u "$WORKER" --claim --max-chunks 64 --batch-size "$S7_TILE_BATCH" --loop --contract FAST_003 --infer-backend "$S7_INFER_BACKEND" >>"$LOG" 2>&1
