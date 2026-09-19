#!/usr/bin/env bash
# Vast side: drain SHARD (default 1/2). Tuned for RTX 3060 Ti 8GB + ~8GiB host RAM.
# float16 + disk memmap + channel-wise finalize; batch 12; max-chunks 1; respawn via loop.
# Override shard for multi-instance fleets: S7_SHARD=2/6 bash tools/run_vast_simple.sh
set -euo pipefail
cd /workspace/magaphragma-connectome

if [[ -f /venv/main/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

python - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("Need CUDA PyTorch on this instance")
print(torch.cuda.get_device_name(0))
PY

pip install -q numpy tifffile 2>/dev/null || true

CKPT=experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt
test -f "$CKPT" || { echo "Missing checkpoint — re-run push_to_vast.sh from your PC"; exit 2; }

SHARD="${S7_SHARD:-1/2}"
WORKER_ID="${S7_WORKER_ID:-vast-$(echo "$SHARD" | tr '/' '-')}"

export S7_WORKER_ID="$WORKER_ID"
export S7_CLAIM_BACKEND="${S7_CLAIM_BACKEND:-local}"
export S7_DURABLE_MODE=local
export S7_TILE_BATCH="${S7_TILE_BATCH:-12}"
export S7_INFER_BACKEND=eager
export S7_OUT=/workspace/s7-out
export S7_STAGE_DIR=/tmp/s7-fast-stage
export S7_ACCUM_DTYPE=float16
export S7_ACCUM_MEMMAP_DIR=/tmp/s7-accum
export S7_STREAM_SAVE=1
export S7_PROGRESS_EVERY=128
export S7_DIRECT_SAVE=1
export MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$S7_OUT" "$S7_STAGE_DIR" "$S7_ACCUM_MEMMAP_DIR"

exec python -u tools/run_affinity_fullvol_s7_fast_worker.py \
  --claim --loop --max-chunks 1 --batch-size "${S7_TILE_BATCH}" \
  --contract FAST_003 --infer-backend eager \
  --shard "$SHARD"
