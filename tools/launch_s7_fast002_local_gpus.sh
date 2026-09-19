#!/usr/bin/env bash
# FAST_002: N processes, one HIP/CUDA device each, shared claim-lock queue.
# Usage: bash tools/launch_s7_fast002_local_gpus.sh [N_GPUS]
set -euo pipefail
N="${1:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export S7_CLAIM_BACKEND="${S7_CLAIM_BACKEND:-local}"
PY="${S7_PYTHON:-python}"

pids=()
cleanup() {
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

for ((i=0; i<N; i++)); do
  export HIP_VISIBLE_DEVICES="$i"
  export CUDA_VISIBLE_DEVICES="$i"
  export S7_WORKER_ID="local-gpu${i}-$$"
  echo "starting worker $S7_WORKER_ID on device $i"
  "$PY" tools/run_affinity_fullvol_s7_fast_worker.py \
    --claim --loop --batch-size 16 --max-chunks 64 --contract FAST_002 \
    > "experiments/phase6e/AFFINITY-FULLVOL-S7-001/worker_gpu${i}.log" 2>&1 &
  pids+=($!)
done

echo "launched ${#pids[@]} workers; waiting (Ctrl-C stops all)"
wait
