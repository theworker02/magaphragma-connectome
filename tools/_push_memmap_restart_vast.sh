#!/usr/bin/env bash
# Push memmap/low-RAM worker + restart Vast with S7_ACCUM_MEMMAP_DIR.
set -euo pipefail
KEY=/tmp/vast_k
cp /mnt/c/Users/matth/.ssh/vast_affinity "$KEY"
chmod 600 "$KEY"
REPO=/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
SSH=(ssh -p 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
SCP=(scp -P 16969 -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)

"${SCP[@]}" "$REPO/tools/run_affinity_fullvol_s7_fast_worker.py" \
  root@ssh3.vast.ai:/workspace/magaphragma-connectome/tools/run_affinity_fullvol_s7_fast_worker.py

"${SSH[@]}" root@ssh3.vast.ai bash -s <<'REMOTE'
set -euo pipefail
pkill -f 'run_affinity_fullvol_s7_fast_worker' 2>/dev/null || true
pkill -f 'vast_worker_loop' 2>/dev/null || true
sleep 2
# drop page cache pressure if possible
sync || true

python3 - <<'PY'
import json
from pathlib import Path
out = Path("/workspace/s7-out")
state = json.loads((out/"queue_state.json").read_text())
n=0
for cid, st in list(state.get("status_by_id", {}).items()):
    if st == "RUNNING" and not (out/"chunks"/cid/"affinities_core_czyx.npy").exists():
        state["status_by_id"][cid] = "NOT_STARTED"
        n += 1
        for p in (out/"claims").glob(f"{cid}.*"):
            try: p.unlink()
            except OSError: pass
(out/"queue_state.json").write_text(json.dumps(state))
print("reset_stale_running", n)
PY

mkdir -p /tmp/s7-accum /workspace/s7-out
cat > /workspace/s7-out/vast_worker_loop.sh <<'LOOP'
#!/usr/bin/env bash
set -uo pipefail
cd /workspace/magaphragma-connectome
[[ -f /venv/main/bin/activate ]] && source /venv/main/bin/activate
export S7_WORKER_ID=vast-rtx3060ti-01
export S7_CLAIM_BACKEND=local
export S7_DURABLE_MODE=local
export S7_ACCUM_DTYPE=float16
export S7_ACCUM_MEMMAP_DIR=/tmp/s7-accum
export S7_STREAM_SAVE=1
export S7_PROGRESS_EVERY=128
export S7_DIRECT_SAVE=1
export MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$S7_OUT" "$S7_STAGE_DIR" "$S7_ACCUM_MEMMAP_DIR"
n=0
while true; do
  n=$((n+1))
  echo "[loop] start attempt=$n $(date -u +%FT%TZ) batch=8 shard=1/2 memmap+stream_save" | tee -a /workspace/s7-out/worker.log
  rm -f /tmp/s7-accum/*.dat 2>/dev/null || true
  python -u tools/run_affinity_fullvol_s7_fast_worker.py \
    --claim --loop --max-chunks 1 --batch-size 8 \
    --contract FAST_003 --infer-backend eager \
    --shard 1/2 >> /workspace/s7-out/worker.log 2>&1
  rc=$?
  echo "[loop] exit rc=$rc attempt=$n $(date -u +%FT%TZ)" | tee -a /workspace/s7-out/worker.log
  rm -f /tmp/s7-accum/*.dat 2>/dev/null || true
  sleep 5
done
LOOP
chmod +x /workspace/s7-out/vast_worker_loop.sh
: > /workspace/s7-out/worker.log
nohup bash /workspace/s7-out/vast_worker_loop.sh >/workspace/s7-out/loop.out 2>&1 &
echo loop_pid=$!
sleep 25
ps aux | grep -E 'run_affinity_fullvol|vast_worker_loop' | grep -v grep
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
free -h
tail -n 40 /workspace/s7-out/worker.log
REMOTE
