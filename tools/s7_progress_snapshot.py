#!/usr/bin/env python3
"""Write Affinity S7 hybrid-drain progress JSON.

Reads queue_state.json + chunk affinity artifacts under S7_OUT (default ~/s7-affinity-out).
Shard 0 = even chunk numeric IDs (local); shard 1 = odd (Vast).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = REPO / "local_research_build/phase5c-production/chunks.json"
DONE_STATUSES = frozenset({"COMPLETE", "AFFINITY_DONE_SEG_PENDING"})
AFFINITY_NAME = "affinities_core_czyx.npy"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def default_out() -> Path:
    env = os.environ.get("S7_OUT")
    if env:
        return Path(env)
    # Prefer durable Linux out (WSL). Avoid Windows FULLVOL fallback when
    # the WSL tree is reachable via \\wsl$ or /home/research-runner.
    candidates = [
        Path.home() / "s7-affinity-out",
        Path("/home/research-runner/s7-affinity-out"),
        Path(r"\\wsl$\Ubuntu-24.04\home\research-runner\s7-affinity-out"),
        Path(os.path.expandvars(r"%USERPROFILE%\s7-affinity-out")),
    ]
    for home in candidates:
        try:
            if home.exists() and ((home / "chunks").is_dir() or (home / "queue_state.json").is_file()):
                return home
        except OSError:
            continue
    return REPO / "experiments/phase6e/AFFINITY-FULLVOL-S7-001"


def chunk_num(cid: str) -> int | None:
    try:
        return int(str(cid).rsplit("-", 1)[-1])
    except ValueError:
        return None


def count_affinity_dirs(chunks_dir: Path) -> tuple[int, int, int]:
    """Return (total, even, odd) dirs that contain affinities_core_czyx.npy."""
    even = odd = total = 0
    if not chunks_dir.is_dir():
        return 0, 0, 0
    for p in chunks_dir.iterdir():
        if not p.is_dir():
            continue
        if not (p / AFFINITY_NAME).exists():
            continue
        total += 1
        n = chunk_num(p.name)
        if n is None:
            continue
        if n % 2 == 0:
            even += 1
        else:
            odd += 1
    return total, even, odd


def snapshot(out: Path, chunks_path: Path) -> dict:
    total = 28798
    if chunks_path.is_file():
        try:
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            total = len(payload.get("chunks") or [])
        except Exception:
            pass

    status_by_id: dict[str, str] = {}
    state_path = out / "queue_state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            status_by_id = dict(state.get("status_by_id") or {})
            if status_by_id:
                total = max(total, len(status_by_id))
        except Exception:
            status_by_id = {}

    counts = dict(Counter(status_by_id.values())) if status_by_id else {}
    queue_done = sum(1 for s in status_by_id.values() if s in DONE_STATUSES)
    running = int(counts.get("RUNNING", 0))
    pending = int(counts.get("NOT_STARTED", 0))
    failed = int(counts.get("FAILED", 0))

    local_even = vast_odd = 0
    for cid, st in status_by_id.items():
        if st not in DONE_STATUSES:
            continue
        n = chunk_num(cid)
        if n is None:
            continue
        if n % 2 == 0:
            local_even += 1
        else:
            vast_odd += 1

    aff_total, aff_even, aff_odd = count_affinity_dirs(out / "chunks")
    # Prefer queue affinity-done (authoritative worker status); fall back to files.
    completed = queue_done if status_by_id else aff_total
    if not status_by_id:
        local_even, vast_odd = aff_even, aff_odd

    pct = round(100.0 * completed / total, 3) if total else 0.0
    claims = 0
    claims_dir = out / "claims"
    if claims_dir.is_dir():
        claims = sum(1 for _ in claims_dir.glob("*.claim.json"))

    return {
        "id": "AFFINITY_S7_HYBRID_PROGRESS",
        "title": "Affinity S7",
        "subtitle": "hybrid drain",
        "updated_at": _now(),
        "out": str(out),
        "total": total,
        "completed": completed,
        "pct": pct,
        "local_even": local_even,
        "vast_odd": vast_odd,
        "running": running,
        "pending": pending,
        "failed": failed,
        "claims": claims,
        "affinity_files": {
            "total": aff_total,
            "even": aff_even,
            "odd": aff_odd,
        },
        "status_counts": counts,
        "shards": {
            "0": {"label": "Local · even", "done": local_even, "role": "local"},
            "1": {"label": "Vast · odd", "done": vast_odd, "role": "vast"},
        },
    }


def write_snapshot(out: Path, chunks_path: Path, dest: Path | None = None) -> dict:
    snap = snapshot(out, chunks_path)
    path = dest or (out / "progress_ui.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snap["_written"] = str(path)
    return snap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None, help="S7_OUT directory")
    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--dest", type=Path, default=None, help="Write path (default: OUT/progress_ui.json)")
    args = ap.parse_args()
    out = args.out or default_out()
    snap = write_snapshot(out, args.chunks, args.dest)
    print(json.dumps(snap, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
