"""Claim-lock for S7 chunk queue: local file lock and HTTP coordinator (Vast+local).

AWS DynamoDB claim backend is DECOMMISSIONED for Affinity production.
Historical DynamoDB helpers remain only as raise-on-call stubs so old scripts fail loud.
"""
from __future__ import annotations

import json
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("S7_OUT", str(REPO / "experiments/phase6e/AFFINITY-FULLVOL-S7-001")))
STATE_PATH = OUT / "queue_state.json"
LOCK_DIR = OUT / "claims"
CHUNKS_PATH = Path(
    os.environ.get(
        "S7_CHUNKS",
        str(REPO / "local_research_build/phase5c-production/chunks.json"),
    )
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def worker_id() -> str:
    return os.environ.get("S7_WORKER_ID") or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def ensure_state() -> dict:
    """Load queue_state.json, creating a fresh NOT_STARTED map if missing."""
    OUT.mkdir(parents=True, exist_ok=True)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))["chunks"]
    state = {
        "id": "AFFINITY_FULLVOL_S7_001",
        "opened_at": _now(),
        "status_by_id": {c["id"]: "NOT_STARTED" for c in chunks},
        "claims": {},
        "completed": [],
        "note": "auto-created on worker host (shard/local fleet)",
    }
    save_state(state)
    print(f"created queue_state at {STATE_PATH} n={len(chunks)}", flush=True)
    return state


def load_state() -> dict:
    return ensure_state()


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(STATE_PATH)


def reclaim_stale_running(state: dict, max_age_s: float = 7200.0) -> None:
    """Return RUNNING without affinities (and optional stale claims) to NOT_STARTED."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    for cid, st in list(state.get("status_by_id", {}).items()):
        if st != "RUNNING":
            continue
        aff = OUT / "chunks" / cid / "affinities_core_czyx.npy"
        claim = LOCK_DIR / f"{cid}.claim.json"
        if aff.exists():
            continue
        if not claim.exists():
            state["status_by_id"][cid] = "NOT_STARTED"
            continue
        try:
            meta = json.loads(claim.read_text(encoding="utf-8"))
            age = time.time() - float(meta.get("claimed_epoch", 0))
            if age > max_age_s:
                state["status_by_id"][cid] = "NOT_STARTED"
                claim.unlink(missing_ok=True)
        except Exception:
            state["status_by_id"][cid] = "NOT_STARTED"


def try_claim_local(chunk_id: str, wid: str | None = None) -> bool:
    """Exclusive claim via O_EXCL claim file + queue_state RUNNING."""
    wid = wid or worker_id()
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    claim_path = LOCK_DIR / f"{chunk_id}.claim.json"
    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # Orphan claim files (crash before status flip) block the shard forever.
        # Reclaim when queue says NOT_STARTED/FAILED and no affinities exist.
        try:
            state = load_state()
            cur = state["status_by_id"].get(chunk_id, "NOT_STARTED")
            aff = OUT / "chunks" / chunk_id / "affinities_core_czyx.npy"
            if cur in {"NOT_STARTED", "FAILED"} and not aff.exists():
                claim_path.unlink(missing_ok=True)
                fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            else:
                return False
        except FileExistsError:
            return False
        except Exception:
            return False
    try:
        payload = {
            "chunk_id": chunk_id,
            "worker_id": wid,
            "claimed_at": _now(),
            "claimed_epoch": time.time(),
        }
        os.write(fd, (json.dumps(payload) + "\n").encode("utf-8"))
    finally:
        os.close(fd)

    state = load_state()
    reclaim_stale_running(state)
    cur = state["status_by_id"].get(chunk_id, "NOT_STARTED")
    if cur not in {"NOT_STARTED", "FAILED"}:
        claim_path.unlink(missing_ok=True)
        return False
    if (OUT / "chunks" / chunk_id / "affinities_core_czyx.npy").exists():
        claim_path.unlink(missing_ok=True)
        state["status_by_id"][chunk_id] = "AFFINITY_DONE_SEG_PENDING"
        save_state(state)
        return False
    state["status_by_id"][chunk_id] = "RUNNING"
    state.setdefault("claims", {})[chunk_id] = {"worker_id": wid, "claimed_at": _now()}
    save_state(state)
    return True


def release_local(chunk_id: str, final_status: str) -> None:
    state = load_state()
    # Never mark complete without affinities on disk
    aff = OUT / "chunks" / chunk_id / "affinities_core_czyx.npy"
    if final_status in {"AFFINITY_DONE_SEG_PENDING", "COMPLETE"} and not aff.exists():
        final_status = "FAILED"
    state["status_by_id"][chunk_id] = final_status
    if final_status in {"AFFINITY_DONE_SEG_PENDING", "COMPLETE"}:
        state.setdefault("completed", [])
        if chunk_id not in state["completed"]:
            state["completed"].append(chunk_id)
    state.get("claims", {}).pop(chunk_id, None)
    save_state(state)
    (LOCK_DIR / f"{chunk_id}.claim.json").unlink(missing_ok=True)


def try_claim_dynamodb(chunk_id: str, table_name: str, wid: str | None = None) -> bool:
    raise RuntimeError(
        "DynamoDB claim backend is DECOMMISSIONED for Affinity production. "
        "Use S7_CLAIM_BACKEND=local or http. See docs/AWS_DECOMMISSION.md and docs/VAST_AFFINITY.md."
    )


def complete_dynamodb(chunk_id: str, table_name: str, final_status: str, artifact_uri: str | None = None) -> None:
    raise RuntimeError(
        "DynamoDB claim backend is DECOMMISSIONED for Affinity production. "
        "Use S7_CLAIM_BACKEND=local or http."
    )


def claim_backend() -> str:
    backend = os.environ.get("S7_CLAIM_BACKEND", "local").strip().lower()
    if backend == "dynamodb":
        raise RuntimeError(
            "S7_CLAIM_BACKEND=dynamodb is DECOMMISSIONED. Use local or http "
            "(python tools/affinity_vast.py claim-server)."
        )
    return backend
