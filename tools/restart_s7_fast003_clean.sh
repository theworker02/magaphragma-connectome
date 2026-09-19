#!/usr/bin/env bash
set -euo pipefail
# Clear stale O_EXCL claim files older than 2h without affinity outputs, then start FAST_003.
ROOT=/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome
OUT=$ROOT/experiments/phase6e/AFFINITY-FULLVOL-S7-001
PY=/opt/venvs/unlearning-rocm/bin/python
export ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.2.1}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib/migraphx/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROCM_PATH}/lib:${PYTHONPATH:-}"
"$PY" <<'PY'
import json, time
from pathlib import Path
OUT = Path("/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome/experiments/phase6e/AFFINITY-FULLVOL-S7-001")
claims = OUT / "claims"
state_path = OUT / "queue_state.json"
state = json.loads(state_path.read_text())
cleared = 0
for claim in claims.glob("*.claim.json"):
    cid = claim.name.replace(".claim.json", "")
    aff = OUT / "chunks" / cid / "affinities_core_czyx.npy"
    if aff.exists():
        continue
    age = time.time() - claim.stat().st_mtime
    if age > 300:  # 5 min
        claim.unlink(missing_ok=True)
        if state.get("status_by_id", {}).get(cid) == "RUNNING":
            state["status_by_id"][cid] = "NOT_STARTED"
        state.get("claims", {}).pop(cid, None)
        cleared += 1
state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
print(f"cleared_stale_claims={cleared}")
PY
exec bash "$ROOT/tools/loop_affinity_fullvol_s7_fast.sh"
