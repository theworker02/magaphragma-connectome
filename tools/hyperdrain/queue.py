"""HyperDrain queue: claim / lease / resume on a dedicated queue_state."""
from __future__ import annotations

import json
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from hyperdrain.config import CHUNKS, OUT, QUEUE_STATE, ROI_PKG, S7_OUT, ensure_out

LOCK_DIR = OUT / "claims"

# States per spec
PENDING = "PENDING"
LEASED = "LEASED"
CLAIMED = "CLAIMED"  # alias
RUNNING = "RUNNING"
VERIFYING = "VERIFYING"
COMPLETE = "COMPLETE"
FAILED_RETRYABLE = "FAILED_RETRYABLE"
RETRYABLE_FAILURE = "FAILED_RETRYABLE"
PERMANENT_FAILURE = "FAILED_PERMANENT"
FAILED_PERMANENT = "FAILED_PERMANENT"
# Compat with S7 affinity-done
AFFINITY_DONE = "AFFINITY_DONE_SEG_PENDING"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def worker_id() -> str:
    return os.environ.get("HYPERDRAIN_WORKER_ID") or f"hd-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def load_state() -> dict:
    ensure_out()
    if not QUEUE_STATE.exists():
        return bootstrap_from_s7()
    return json.loads(QUEUE_STATE.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    ensure_out()
    tmp = QUEUE_STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(QUEUE_STATE)


def bootstrap_from_s7() -> dict:
    """Initialize HyperDrain queue from chunk list; skip S7-complete ids."""
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))["chunks"]
    s7_state = {}
    s7_path = S7_OUT / "queue_state.json"
    if s7_path.exists():
        s7_state = json.loads(s7_path.read_text(encoding="utf-8")).get("status_by_id", {})
    status = {}
    completed = []
    for c in chunks:
        cid = c["id"]
        prior = s7_state.get(cid, "NOT_STARTED")
        aff_s7 = S7_OUT / "chunks" / cid / "affinities_core_czyx.npy"
        aff_hd = OUT / "chunks" / cid / "affinities_core_czyx.npy"
        if aff_hd.exists() or aff_s7.exists() or prior in {"AFFINITY_DONE_SEG_PENDING", "COMPLETE"}:
            status[cid] = COMPLETE if aff_hd.exists() or prior == "COMPLETE" else AFFINITY_DONE
            completed.append(cid)
        else:
            status[cid] = PENDING
    state = {
        "created_at": _now(),
        "engine": "hyperdrain",
        "status_by_id": status,
        "completed": completed,
        "leases": {},
    }
    save_state(state)
    return state


def heartbeat(chunk_id: str, wid: str) -> None:
    state = load_state()
    state.setdefault("leases", {})[chunk_id] = {
        "worker_id": wid,
        "heartbeat_at": _now(),
        "epoch": time.time(),
    }
    save_state(state)
    claim = LOCK_DIR / f"{chunk_id}.claim.json"
    if claim.exists():
        try:
            meta = json.loads(claim.read_text(encoding="utf-8"))
            meta["heartbeat_at"] = _now()
            meta["epoch"] = time.time()
            claim.write_text(json.dumps(meta) + "\n", encoding="utf-8")
        except Exception:
            pass


def reclaim_stale(state: dict, max_age_s: float = 7200.0) -> None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    for cid, st in list(state.get("status_by_id", {}).items()):
        if st not in {RUNNING, CLAIMED, LEASED, VERIFYING}:
            continue
        aff = OUT / "chunks" / cid / "affinities_core_czyx.npy"
        if aff.exists():
            state["status_by_id"][cid] = AFFINITY_DONE
            continue
        lease = state.get("leases", {}).get(cid) or {}
        claim = LOCK_DIR / f"{cid}.claim.json"
        epoch = float(lease.get("epoch") or 0)
        if claim.exists():
            try:
                epoch = max(epoch, float(json.loads(claim.read_text()).get("epoch", 0)))
            except Exception:
                pass
        if epoch and (time.time() - epoch) > max_age_s:
            state["status_by_id"][cid] = PENDING
            state.get("leases", {}).pop(cid, None)
            claim.unlink(missing_ok=True)
        elif not epoch and not claim.exists():
            state["status_by_id"][cid] = PENDING


def try_claim(chunk_id: str, wid: str | None = None) -> bool:
    wid = wid or worker_id()
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    claim_path = LOCK_DIR / f"{chunk_id}.claim.json"
    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        os.write(
            fd,
            (json.dumps({"chunk_id": chunk_id, "worker_id": wid, "claimed_at": _now(), "epoch": time.time()}) + "\n").encode(),
        )
    finally:
        os.close(fd)

    state = load_state()
    reclaim_stale(state)
    cur = state["status_by_id"].get(chunk_id, PENDING)
    if cur not in {PENDING, FAILED_RETRYABLE}:
        claim_path.unlink(missing_ok=True)
        return False
    if (OUT / "chunks" / chunk_id / "affinities_core_czyx.npy").exists():
        claim_path.unlink(missing_ok=True)
        state["status_by_id"][chunk_id] = AFFINITY_DONE
        save_state(state)
        return False
    state["status_by_id"][chunk_id] = LEASED
    # RUNNING set when worker begins inference
    state["status_by_id"][chunk_id] = RUNNING
    state.setdefault("leases", {})[chunk_id] = {"worker_id": wid, "heartbeat_at": _now(), "epoch": time.time()}
    save_state(state)
    return True


def complete(chunk_id: str, status: str = AFFINITY_DONE) -> None:
    state = load_state()
    state["status_by_id"][chunk_id] = status
    if status in {AFFINITY_DONE, COMPLETE}:
        state.setdefault("completed", [])
        if chunk_id not in state["completed"]:
            state["completed"].append(chunk_id)
    state.get("leases", {}).pop(chunk_id, None)
    save_state(state)
    (LOCK_DIR / f"{chunk_id}.claim.json").unlink(missing_ok=True)


def fail(chunk_id: str, permanent: bool = False, error: str = "") -> None:
    state = load_state()
    state["status_by_id"][chunk_id] = FAILED_PERMANENT if permanent else FAILED_RETRYABLE
    state.setdefault("failures", []).append({"id": chunk_id, "error": error, "at": _now()})
    state.get("leases", {}).pop(chunk_id, None)
    save_state(state)
    (LOCK_DIR / f"{chunk_id}.claim.json").unlink(missing_ok=True)


def pending_chunks(chunks: list[dict]) -> list[dict]:
    state = load_state()
    reclaim_stale(state)
    save_state(state)
    out = []
    for c in chunks:
        st = state["status_by_id"].get(c["id"], PENDING)
        if st in {PENDING, FAILED_RETRYABLE}:
            out.append(c)
    return out


def status_summary() -> dict:
    state = load_state()
    from collections import Counter

    counts = Counter(state.get("status_by_id", {}).values())
    n = sum(counts.values()) or 1
    done = counts.get(COMPLETE, 0) + counts.get(AFFINITY_DONE, 0)
    return {
        "counts": dict(counts),
        "n_total": n,
        "n_done": done,
        "fraction_done": done / n,
        "engine": "hyperdrain",
        "updated_at": _now(),
    }


def register_worker(meta: dict) -> None:
    state = load_state()
    wid = meta.get("worker_id") or worker_id()
    workers = state.setdefault("workers", {})
    workers[wid] = {**meta, "last_heartbeat": _now()}
    save_state(state)


def fleet_metrics(recent_tiles_per_sec: float | None = None) -> dict:
    state = load_state()
    summary = status_summary()
    workers = state.get("workers") or {}
    online = [w for w in workers.values() if (time.time() - float(w.get("epoch", 0) or 0)) < 120]
    # fallback: count RUNNING leases
    if not online:
        online = list((state.get("leases") or {}).values())
    n_gpu = max(1, len(online)) if online else 1
    # rolling from recent chunk telemetries if present
    tps = float(recent_tiles_per_sec or 0.0)
    if tps <= 0:
        cfg_path = OUT / "HYPERDRAIN_PACKED_CONFIG.json"
        if cfg_path.exists():
            try:
                best = json.loads(cfg_path.read_text(encoding="utf-8")).get("best") or {}
                tps = float(best.get("tiles_per_second") or 0.0) * max(1, len(online) or 1)
            except Exception:
                tps = 0.0
        if tps <= 0:
            from hyperdrain.config import BASELINE_EAGER_TILES_PER_SEC
            tps = BASELINE_EAGER_TILES_PER_SEC * max(1, len(online) or 1)
    # representative 20736 tiles/chunk
    tiles_per_chunk = 20736
    chunks_per_hour = (tps * 3600.0) / tiles_per_chunk if tps > 0 else 0.0
    remaining = max(0, summary["n_total"] - summary["n_done"])
    eta_hours = remaining / chunks_per_hour if chunks_per_hour > 0 else None
    return {
        "gpus_online": len(online),
        "workers": len(workers),
        "aggregate_tiles_per_second": tps,
        "aggregate_chunks_per_hour": chunks_per_hour,
        "chunks_remaining": remaining,
        "eta_hours": eta_hours,
        "effective_GPU_count": n_gpu,
    }


def format_mass_status() -> str:
    s = status_summary()
    f = fleet_metrics()
    qpath = OUT / "HYPERDRAIN_PRODUCTION_BACKEND.json"
    backend = "eager"
    pack = "?"
    equiv = "?"
    if qpath.exists():
        import json as _json
        q = _json.loads(qpath.read_text(encoding="utf-8"))
        backend = q.get("production", backend)
        equiv = "PASS" if q.get("mass_production_speedup_validated") or q.get("speedup_validated") else "pending"
    cfg = OUT / "HYPERDRAIN_PACKED_CONFIG.json"
    if cfg.exists():
        import json as _json
        pack = _json.loads(cfg.read_text(encoding="utf-8")).get("pack_size", pack)
    eta = f.get("eta_hours")
    if eta is None:
        eta_s = "n/a"
    else:
        d = int(eta // 24)
        h = int(eta % 24)
        eta_s = f"{d}d {h}h"
    counts = s.get("counts") or {}
    return (
        "HYPERDRAIN MASS PRODUCTION\n\n"
        f"Chunks\n"
        f"  Complete:       {s['n_done']:,} / {s['n_total']:,}\n"
        f"  Running:        {counts.get('RUNNING', 0)}\n"
        f"  Pending:        {counts.get('PENDING', 0) + counts.get('FAILED_RETRYABLE', 0)}\n"
        f"  Failed:         {counts.get('FAILED_PERMANENT', 0)}\n\n"
        f"Fleet\n"
        f"  GPUs online:    {f['gpus_online']}\n"
        f"  Workers:        {f['workers']}\n\n"
        f"Performance\n"
        f"  Aggregate:      {f['aggregate_tiles_per_second']:.1f} tiles/s\n"
        f"  Chunks/hour:    {f['aggregate_chunks_per_hour']:.1f}\n\n"
        f"ETA\n"
        f"  Remaining:      {eta_s}\n\n"
        f"Backend\n"
        f"  {backend}\n"
        f"  pack={pack}\n"
        f"  mass_speedup: {equiv}\n"
    )