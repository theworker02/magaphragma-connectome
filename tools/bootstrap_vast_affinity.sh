#!/usr/bin/env bash
# Bootstrap a Vast.ai CUDA Jupyter host into an Affinity S7 production worker.
# Usage (on Vast /workspace):
#   bash tools/bootstrap_vast_affinity.sh
# Required env before production:
#   S7_CLAIM_HTTP=http://CONTROLLER:8787
#   S7_WORKER_ID=vast-rtx3060ti-01
#   S7_DURABLE_MODE=http   # upload artifacts to coordinator before complete
set -euo pipefail

ROOT="${AFFINITY_ROOT:-/workspace/magaphragma-connectome}"
VENV="${AFFINITY_VENV:-/venv/main}"
WORKER_ID="${S7_WORKER_ID:-vast-rtx3060ti-01}"
CLAIM_HTTP="${S7_CLAIM_HTTP:?Set S7_CLAIM_HTTP to controller claim URL e.g. http://x.x.x.x:8787}"
EXPECTED_SHA256="73ad8a06229127b6eb9c759e7ba208a3c6a70c2b3556d7742cb6e7d8bd292b92"
CKPT_REL="experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt"
AUTH_FLAG="/tmp/affinity_production_authorized"

echo "=== Affinity Vast bootstrap ==="
echo "ROOT=$ROOT WORKER_ID=$WORKER_ID CLAIM_HTTP=$CLAIM_HTTP"

# 1. GPU
command -v nvidia-smi >/dev/null || { echo "FAIL: nvidia-smi missing"; exit 2; }
nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "GPU=$GPU_NAME VRAM_MB=$VRAM_MB"
if [[ "${VRAM_MB:-0}" -lt 6000 ]]; then
  echo "FAIL: need >= 6GB VRAM"
  exit 2
fi

# 2. Repo
if [[ ! -d "$ROOT/.git" && ! -f "$ROOT/tools/run_affinity_fullvol_s7_fast_worker.py" ]]; then
  echo "Repo not found at $ROOT"
  echo "Deploy via rsync/scp from your PC (preferred; no GitHub creds on Vast):"
  echo "  rsync -avz --exclude .venv --exclude .git/objects \\"
  echo "    ./magaphragma-connectome/ vast:/workspace/magaphragma-connectome/"
  exit 2
fi
cd "$ROOT"

# 3. Python / PyTorch CUDA (do NOT use ROCm)
# shellcheck disable=SC1091
if [[ -f "$VENV/bin/activate" ]]; then
  # Prefer Vast-provided venv
  source "$VENV/bin/activate"
else
  python3 -m venv /workspace/venv-affinity
  source /workspace/venv-affinity/bin/activate
  pip install -U pip
fi

python - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    sys.exit("FAIL: PyTorch CUDA not available — install a CUDA wheel matching the host driver")
print("device", torch.cuda.get_device_name(0), flush=True)
PY

pip install -q numpy tifffile requests 2>/dev/null || pip install numpy tifffile requests

# 4. Checkpoint
CKPT="$ROOT/$CKPT_REL"
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint missing: $CKPT"
  echo "From LOCAL WSL, copy it:"
  echo "  scp -P \$VAST_SSH_PORT $CKPT_REL root@\$VAST_SSH_HOST:/workspace/magaphragma-connectome/$CKPT_REL"
  exit 2
fi
GOT=$(sha256sum "$CKPT" | awk '{print $1}')
if [[ "$GOT" != "$EXPECTED_SHA256" ]]; then
  echo "FAIL: checkpoint sha256 mismatch"
  echo " got  $GOT"
  echo " want $EXPECTED_SHA256"
  exit 2
fi
echo "checkpoint sha256 OK"

# 5. Queue connectivity
python - <<PY
from urllib.request import urlopen
import json, sys
url = "${CLAIM_HTTP}".rstrip("/") + "/health"
try:
    data = json.loads(urlopen(url, timeout=20).read().decode())
except Exception as e:
    print("FAIL: claim coordinator unreachable:", e)
    sys.exit(2)
print("claim health", data)
if not data.get("ok"):
    sys.exit(2)
PY

# 6. Qualification (must pass before production)
export S7_CHECKPOINT="$CKPT"
export S7_QUAL_OUT="$ROOT/experiments/phase6e/VAST-AFFINITY-BENCH-001"
export S7_PRODUCTION_AUTH_FLAG="$AUTH_FLAG"
rm -f "$AUTH_FLAG"
python -u tools/qualify_nvidia_affinity.py --batches 16,24,32,48 --backend eager
test -f "$AUTH_FLAG" || { echo "FAIL: qualification did not authorize production"; exit 3; }

BEST_BATCH=$(python -c "import json; print(json.load(open('$AUTH_FLAG'))['best']['batch'])")
echo "Qualified best_batch=$BEST_BATCH"

# 7. Register worker + start production
export S7_WORKER_ID="$WORKER_ID"
export S7_CLAIM_BACKEND=http
export S7_CLAIM_HTTP="$CLAIM_HTTP"
export S7_DURABLE_MODE=http
export S7_TILE_BATCH="${S7_TILE_BATCH:-$BEST_BATCH}"
export S7_INFER_BACKEND=eager
export S7_OUT="${S7_OUT:-/workspace/s7-out}"
export S7_STAGE_DIR="${S7_STAGE_DIR:-/tmp/s7-fast-stage}"
export S7_CHUNKS="${S7_CHUNKS:-$ROOT/local_research_build/phase5c-production/chunks.json}"
mkdir -p "$S7_OUT" "$S7_STAGE_DIR"

python - <<PY
from cloud.claim_http import register_worker_http
import os
register_worker_http(os.environ["S7_CLAIM_HTTP"], {
    "worker_id": os.environ["S7_WORKER_ID"],
    "gpu": "${GPU_NAME}",
    "runtime": "CUDA",
    "batch": int(os.environ["S7_TILE_BATCH"]),
    "backend": "eager",
    "dph_usd": float(os.environ.get("S7_VAST_DPH", "0") or 0),
    "instance_id": os.environ.get("S7_VAST_INSTANCE_ID", ""),
})
print("registered", os.environ["S7_WORKER_ID"])
PY

echo "=== Starting production worker (claim+loop) ==="
exec python -u tools/run_affinity_fullvol_s7_fast_worker.py \
  --claim --loop --max-chunks 64 \
  --batch-size "$S7_TILE_BATCH" \
  --contract FAST_003 \
  --infer-backend eager
