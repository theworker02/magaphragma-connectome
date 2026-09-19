"""Provider-neutral HTTP claim + artifact coordinator for Affinity fleets.

Single-controller design (runs on the machine that owns AFFINITY-FULLVOL-S7-001):

Atomicity
---------
Claims use ``os.O_EXCL`` create of ``claims/<chunk_id>.claim.json`` on the
controller filesystem (via ``try_claim_local``). Only one process can create
that inode. Concurrent HTTP ``/claim`` calls therefore cannot both succeed for
the same chunk — the second gets ``ok: false``.

Endpoints
---------
GET  /health
GET  /status
POST /claim      {chunk_id, worker_id}
POST /heartbeat  {chunk_id, worker_id}
POST /complete   {chunk_id, status}  — refuses COMPLETE without durable affinities
POST /reclaim-stale
POST /next       {worker_id} -> {chunk_id} or null  (claim next NOT_STARTED)
POST /register-worker {worker_id, gpu, runtime, meta}
POST /telemetry  {worker_id, ...}
PUT  /artifact/<chunk_id>/<filename>  — stream durable bytes before complete
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from s7_chunk_claim import (  # noqa: E402
    LOCK_DIR,
    OUT,
    load_state,
    reclaim_stale_running,
    release_local,
    save_state,
    try_claim_local,
    worker_id,
)

WORKERS_PATH = OUT / "fleet_workers.json"
TELEMETRY_PATH = OUT / "fleet_telemetry.json"
_lock = threading.Lock()
EXPECTED_AFFINITY_NAMES = ("affinities_core_czyx.npy", "boundaries_core.tif", "receipt.json")


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def durable_affinities_exist(chunk_id: str) -> bool:
    aff = OUT / "chunks" / chunk_id / "affinities_core_czyx.npy"
    return aff.is_file() and aff.stat().st_size > 0


# ----- client helpers (used by Vast / remote workers) -----


def try_claim_http(chunk_id: str, base_url: str, wid: str | None = None) -> bool:
    wid = wid or worker_id()
    url = base_url.rstrip("/") + "/claim"
    body = json.dumps({"chunk_id": chunk_id, "worker_id": wid}).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return bool(data.get("ok"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False


def complete_http(chunk_id: str, base_url: str, final_status: str) -> None:
    url = base_url.rstrip("/") + "/complete"
    body = json.dumps({"chunk_id": chunk_id, "status": final_status}).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        if not data.get("ok"):
            raise RuntimeError(f"complete_http failed: {data}")


def heartbeat_http(chunk_id: str, base_url: str, wid: str | None = None) -> bool:
    wid = wid or worker_id()
    url = base_url.rstrip("/") + "/heartbeat"
    body = json.dumps({"chunk_id": chunk_id, "worker_id": wid}).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=30) as resp:
            return bool(json.loads(resp.read().decode("utf-8")).get("ok"))
    except Exception:
        return False


def next_claim_http(base_url: str, wid: str | None = None) -> str | None:
    wid = wid or worker_id()
    url = base_url.rstrip("/") + "/next"
    body = json.dumps({"worker_id": wid}).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("chunk_id")


def upload_artifact_http(base_url: str, chunk_id: str, local_path: Path, remote_name: str | None = None) -> dict:
    """Stream a file to the coordinator durable store (must finish before /complete)."""
    name = remote_name or local_path.name
    url = f"{base_url.rstrip('/')}/artifact/{chunk_id}/{name}"
    size = local_path.stat().st_size

    class _FileStream:
        def __init__(self, path: Path, n: int) -> None:
            self._f = path.open("rb")
            self.len = n

        def read(self, n: int = -1) -> bytes:
            return self._f.read(n if n is not None and n >= 0 else None)

        def __len__(self) -> int:
            return self.len

        def close(self) -> None:
            self._f.close()

    stream = _FileStream(local_path, size)
    try:
        req = Request(
            url,
            data=stream,  # type: ignore[arg-type]
            method="PUT",
            headers={"Content-Type": "application/octet-stream", "Content-Length": str(size)},
        )
        with urlopen(req, timeout=7200) as resp:
            return json.loads(resp.read().decode("utf-8"))
    finally:
        stream.close()


def register_worker_http(base_url: str, payload: dict) -> None:
    url = base_url.rstrip("/") + "/register-worker"
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        if not data.get("ok"):
            raise RuntimeError(data)


def telemetry_http(base_url: str, payload: dict) -> None:
    url = base_url.rstrip("/") + "/telemetry"
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        urlopen(req, timeout=15).read()
    except Exception:
        pass


def _claim_next(wid: str) -> str | None:
    """Atomically claim the next NOT_STARTED chunk (controller-local)."""
    state = load_state()
    reclaim_stale_running(state)
    save_state(state)
    # Prefer chunks already ordered in state if present; else scan status map
    candidates = [
        cid
        for cid, st in state.get("status_by_id", {}).items()
        if st == "NOT_STARTED"
    ]
    candidates.sort()
    for cid in candidates:
        if try_claim_local(cid, wid=wid):
            return cid
    return None


def _heartbeat(chunk_id: str, wid: str) -> bool:
    claim = LOCK_DIR / f"{chunk_id}.claim.json"
    if not claim.exists():
        return False
    try:
        meta = json.loads(claim.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("worker_id") and meta.get("worker_id") != wid:
        return False
    meta["heartbeat_at"] = _now()
    meta["heartbeat_epoch"] = time.time()
    claim.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return True


def _queue_counts() -> dict[str, int]:
    state = load_state()
    counts: dict[str, int] = {}
    for st in state.get("status_by_id", {}).values():
        counts[st] = counts.get(st, 0) + 1
    return counts


class ClaimHandler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            self._json(200, {"ok": True, "coordinator": "affinity-s7", "out": str(OUT)})
            return
        if path == "/status":
            with _lock:
                counts = _queue_counts()
                workers = _load_json(WORKERS_PATH, {})
                telem = _load_json(TELEMETRY_PATH, {})
            self._json(
                200,
                {
                    "ok": True,
                    "counts": counts,
                    "n_total": sum(counts.values()),
                    "workers": workers,
                    "telemetry": telem,
                    "updated_at": _now(),
                },
            )
            return
        self._json(404, {"ok": False, "error": "not_found"})

    def do_PUT(self) -> None:  # noqa: N802
        # /artifact/<chunk_id>/<filename>
        parts = self.path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "artifact":
            self._json(404, {"ok": False, "error": "not_found"})
            return
        chunk_id, filename = parts[1], parts[2]
        if ".." in chunk_id or ".." in filename or "/" in filename or "\\" in filename:
            self._json(400, {"ok": False, "error": "bad_path"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            self._json(400, {"ok": False, "error": "empty"})
            return
        # Cap absurd uploads (affinities ~1.6GiB; allow up to 4GiB)
        if n > 4 * 1024**3:
            self._json(413, {"ok": False, "error": "too_large"})
            return
        dest_dir = OUT / "chunks" / chunk_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest_dir / (filename + ".partial")
        remaining = n
        with tmp.open("wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        if remaining != 0:
            tmp.unlink(missing_ok=True)
            self._json(400, {"ok": False, "error": "short_read"})
            return
        final = dest_dir / filename
        tmp.replace(final)
        self._json(200, {"ok": True, "path": str(final), "bytes": n})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            self._json(400, {"ok": False, "error": "bad_json"})
            return

        with _lock:
            if path == "/claim":
                cid = str(body.get("chunk_id") or "")
                wid = str(body.get("worker_id") or "")
                if not cid:
                    self._json(400, {"ok": False, "error": "chunk_id required"})
                    return
                ok = try_claim_local(cid, wid=wid or None)
                self._json(200, {"ok": ok, "chunk_id": cid, "worker_id": wid})
                return

            if path == "/next":
                wid = str(body.get("worker_id") or worker_id())
                cid = _claim_next(wid)
                self._json(200, {"ok": cid is not None, "chunk_id": cid, "worker_id": wid})
                return

            if path == "/heartbeat":
                cid = str(body.get("chunk_id") or "")
                wid = str(body.get("worker_id") or "")
                ok = _heartbeat(cid, wid) if cid and wid else False
                self._json(200, {"ok": ok})
                return

            if path == "/complete":
                cid = str(body.get("chunk_id") or "")
                status = str(body.get("status") or "AFFINITY_DONE_SEG_PENDING")
                if not cid:
                    self._json(400, {"ok": False, "error": "chunk_id required"})
                    return
                if status in {"AFFINITY_DONE_SEG_PENDING", "COMPLETE"} and not durable_affinities_exist(cid):
                    self._json(
                        409,
                        {
                            "ok": False,
                            "error": "affinities missing on durable controller store — upload artifacts before complete",
                            "chunk_id": cid,
                        },
                    )
                    return
                release_local(cid, status)
                self._json(200, {"ok": True, "chunk_id": cid, "status": status})
                return

            if path == "/reclaim-stale":
                max_age = float(body.get("max_age_s") or 7200)
                state = load_state()
                before = dict(state.get("status_by_id", {}))
                reclaim_stale_running(state, max_age_s=max_age)
                save_state(state)
                reclaimed = [
                    cid
                    for cid, st in before.items()
                    if st == "RUNNING" and state["status_by_id"].get(cid) == "NOT_STARTED"
                ]
                self._json(200, {"ok": True, "reclaimed": reclaimed, "n": len(reclaimed)})
                return

            if path == "/register-worker":
                wid = str(body.get("worker_id") or "")
                if not wid:
                    self._json(400, {"ok": False, "error": "worker_id required"})
                    return
                workers = _load_json(WORKERS_PATH, {})
                workers[wid] = {**body, "registered_at": _now(), "last_seen": _now()}
                _save_json(WORKERS_PATH, workers)
                self._json(200, {"ok": True})
                return

            if path == "/telemetry":
                wid = str(body.get("worker_id") or "unknown")
                telem = _load_json(TELEMETRY_PATH, {})
                telem[wid] = {**body, "at": _now()}
                _save_json(TELEMETRY_PATH, telem)
                workers = _load_json(WORKERS_PATH, {})
                if wid in workers:
                    workers[wid]["last_seen"] = _now()
                    _save_json(WORKERS_PATH, workers)
                self._json(200, {"ok": True})
                return

        self._json(404, {"ok": False, "error": "not_found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("claim_http: " + (fmt % args) + "\n")


def serve(host: str = "0.0.0.0", port: int = 8787) -> None:
    # Conservative reclaim of stale RUNNING on boot
    state = load_state()
    reclaim_stale_running(state, max_age_s=float(os.environ.get("S7_CLAIM_MAX_AGE_S", "7200")))
    save_state(state)
    httpd = ThreadingHTTPServer((host, port), ClaimHandler)
    print(
        f"Affinity claim coordinator on http://{host}:{port} out={OUT} "
        f"(O_EXCL claims — dual success impossible)",
        flush=True,
    )
    httpd.serve_forever()


def main() -> int:
    host = os.environ.get("S7_CLAIM_HTTP_HOST", "0.0.0.0")
    port = int(os.environ.get("S7_CLAIM_HTTP_PORT", "8787"))
    serve(host, port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
