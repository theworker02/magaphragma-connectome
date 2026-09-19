#!/usr/bin/env bash
# Push low-RAM worker patch to Vast and restart shard 1/2 with float16 accum.
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
REPO=/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
SSH=(ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
SCP=(scp -P 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)

echo "=== push worker ==="
"${SCP[@]}" "$REPO/tools/run_affinity_fullvol_s7_fast_worker.py" \
  root@ssh3.vast.ai:/workspace/magaphragma-connectome/tools/run_affinity_fullvol_s7_fast_worker.py

"${SSH[@]}" root@ssh3.vast.ai bash -s <<'REMOTE'
set -euo pipefail
pkill -f 'run_affinity_fullvol_s7_fast_worker' 2>/dev/null || true
pkill -f 'vast_worker_loop' 2>/dev/null || true
sleep 2

# reset stale RUNNING without affinities
python3 - <<'PY'
import json
from pathlib import Path
out = Path("/workspace/s7-out")
state_path = out / "queue_state.json"
state = json.loads(state_path.read_text())
n = 0
for cid, st in list(state.get("status_by_id", {}).items()):
    if st == "RUNNING":
        aff = out / "chunks" / cid / "affinities_core_czyx.npy"
        if not aff.exists():
            state["status_by_id"][cid] = "NOT_STARTED"
            n += 1
            for p in (out / "claims").glob(f"{cid}.*"):
                try: p.unlink()
                except OSError: pass
state_path.write_text(json.dumps(state))
print(f"reset_stale_running={n}")
PY

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
export S7_ACCUM_DTYPE=float16
export S7_PROGRESS_EVERY=64
export MALLOC_ARENA_MAX=2
mkdir -p "$S7_OUT" "$S7_STAGE_DIR"
n=0
while true; do
  n=$((n+1))
  echo "[loop] start attempt=$n $(date -u +%FT%TZ) batch=8 shard=1/2 accum=float16" | tee -a /workspace/s7-out/worker.log
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
: > /workspace/s7-out/worker.log
nohup bash /workspace/s7-out/vast_worker_loop.sh >/workspace/s7-out/loop.out 2>&1 &
echo "loop_pid=$!"
sleep 20
echo "=== processes ==="
ps aux | grep -E 'run_affinity_fullvol|vast_worker_loop' | grep -v grep
echo "=== nvidia ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
echo "=== free ==="
free -h
echo "=== log ==="
tail -n 50 /workspace/s7-out/worker.log
REMOTE
