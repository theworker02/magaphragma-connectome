#!/usr/bin/env bash
# Stabilize Vast Affinity worker: swap for RAM spikes + respawn loop + shard 1/2.
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new root@ssh3.vast.ai bash -s <<'REMOTE'
set -euo pipefail

echo "=== kill old workers ==="
pkill -f 'run_affinity_fullvol_s7_fast_worker' 2>/dev/null || true
pkill -f 'run_vast_simple.sh' 2>/dev/null || true
pkill -f 'vast_worker_loop' 2>/dev/null || true
sleep 2

echo "=== ensure swap (16G) ==="
if ! swapon --show | grep -q /swapfile; then
  if [[ ! -f /swapfile ]]; then
    fallocate -l 16G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=16384
    chmod 600 /swapfile
    mkswap /swapfile
  fi
  swapon /swapfile || true
fi
swapon --show
free -h

echo "=== reset stale RUNNING claims without affinities ==="
python3 - <<'PY'
import json
from pathlib import Path
out = Path("/workspace/s7-out")
state_path = out / "queue_state.json"
if not state_path.exists():
    print("no queue_state")
    raise SystemExit(0)
state = json.loads(state_path.read_text())
n = 0
for cid, st in list(state.get("status_by_id", {}).items()):
    if st != "RUNNING":
        continue
    aff = out / "chunks" / cid / "affinities_core_czyx.npy"
    if not aff.exists():
        state["status_by_id"][cid] = "NOT_STARTED"
        n += 1
        # drop stale claim file if present
        for p in (out / "claims").glob(f"{cid}.*"):
            try:
                p.unlink()
            except OSError:
                pass
state_path.write_text(json.dumps(state))
print(f"reset_stale_running={n}")
PY

mkdir -p /workspace/s7-out
cat > /workspace/s7-out/vast_worker_loop.sh <<'LOOP'
#!/usr/bin/env bash
set -uo pipefail
cd /workspace/magaphragma-connectome
export S7_WORKER_ID=vast-rtx3060ti-01
export S7_CLAIM_BACKEND=local
export S7_DURABLE_MODE=local
export S7_TILE_BATCH=8
export S7_INFER_BACKEND=eager
export S7_OUT=/workspace/s7-out
export S7_STAGE_DIR=/tmp/s7-fast-stage
# curb glibc arena bloat on small RAM hosts
export MALLOC_ARENA_MAX=2
mkdir -p "$S7_OUT" "$S7_STAGE_DIR"
n=0
while true; do
  n=$((n+1))
  echo "[loop] start attempt=$n $(date -u +%FT%TZ)" >> /workspace/s7-out/worker.log
  bash tools/run_vast_simple.sh >> /workspace/s7-out/worker.log 2>&1
  rc=$?
  echo "[loop] exit rc=$rc attempt=$n $(date -u +%FT%TZ)" >> /workspace/s7-out/worker.log
  sleep 5
done
LOOP
# Override tile batch inside run_vast_simple by wrapping python env;
# run_vast_simple hardcodes --batch-size 16 — patch env S7_TILE_BATCH is used as default
# but CLI --batch-size 16 wins. Rewrite a thin runner instead:
cat > /workspace/s7-out/vast_worker_loop.sh <<'LOOP'
#!/usr/bin/env bash
set -uo pipefail
cd /workspace/magaphragma-connectome
if [[ -f /venv/main/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi
export S7_WORKER_ID=vast-rtx3060ti-01
export S7_CLAIM_BACKEND=local
export S7_DURABLE_MODE=local
export S7_TILE_BATCH=8
export S7_INFER_BACKEND=eager
export S7_OUT=/workspace/s7-out
export S7_STAGE_DIR=/tmp/s7-fast-stage
export MALLOC_ARENA_MAX=2
mkdir -p "$S7_OUT" "$S7_STAGE_DIR"
CKPT=experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt
test -f "$CKPT" || { echo "Missing checkpoint"; exit 2; }
n=0
while true; do
  n=$((n+1))
  echo "[loop] start attempt=$n $(date -u +%FT%TZ) batch=8 shard=1/2" | tee -a /workspace/s7-out/worker.log
  python -u tools/run_affinity_fullvol_s7_fast_worker.py \
    --claim --loop --max-chunks 8 --batch-size 8 \
    --contract FAST_003 --infer-backend eager \
    --shard 1/2 >> /workspace/s7-out/worker.log 2>&1
  rc=$?
  echo "[loop] exit rc=$rc attempt=$n $(date -u +%FT%TZ)" | tee -a /workspace/s7-out/worker.log
  sleep 8
done
LOOP
chmod +x /workspace/s7-out/vast_worker_loop.sh

# truncate old log but keep a bak
cp /workspace/s7-out/worker.log /workspace/s7-out/worker.log.bak 2>/dev/null || true
: > /workspace/s7-out/worker.log

nohup bash /workspace/s7-out/vast_worker_loop.sh >/workspace/s7-out/loop.out 2>&1 &
echo "loop_pid=$!"
sleep 15
echo "=== processes ==="
ps aux | grep -E 'run_affinity_fullvol|vast_worker_loop' | grep -v grep
echo "=== free ==="
free -h
echo "=== nvidia ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
echo "=== log ==="
tail -n 40 /workspace/s7-out/worker.log
REMOTE
