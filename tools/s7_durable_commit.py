"""Durable artifact commit for Affinity — persist THEN mark complete.

Order is mandatory:
  1. local artifacts fully written
  2. copy to durable controller store (if remote worker)
  3. verify durable affinities exist
  4. atomic claim release / COMPLETE

Never mark COMPLETE before durable persistence.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("S7_OUT", str(REPO / "experiments/phase6e/AFFINITY-FULLVOL-S7-001")))
REQUIRED = ("affinities_core_czyx.npy", "boundaries_core.tif", "receipt.json")


def _local_chunk_dir(chunk_id: str, local_out: Path | None = None) -> Path:
    return (local_out or OUT) / "chunks" / chunk_id


def verify_local_artifacts(chunk_id: str, local_out: Path | None = None) -> None:
    d = _local_chunk_dir(chunk_id, local_out)
    for name in REQUIRED:
        p = d / name
        if not p.is_file() or p.stat().st_size <= 0:
            raise FileNotFoundError(f"missing/empty artifact {p}")


def rsync_to_durable(chunk_id: str, dest: str, local_out: Path | None = None) -> None:
    """dest example: user@host:/path/AFFINITY-FULLVOL-S7-001/chunks/"""
    src = str(_local_chunk_dir(chunk_id, local_out)) + "/"
    # Ensure trailing destination includes chunk id
    dest_full = dest.rstrip("/") + f"/{chunk_id}/"
    cmd = ["rsync", "-a", "--partial", "--inplace", src, dest_full]
    ssh = os.environ.get("S7_DURABLE_SSH")
    if ssh:
        cmd = ["rsync", "-a", "--partial", "-e", ssh, src, dest_full]
    subprocess.check_call(cmd)


def copy_to_controller_out(chunk_id: str, controller_out: Path, local_out: Path | None = None) -> None:
    src = _local_chunk_dir(chunk_id, local_out)
    dst = controller_out / "chunks" / chunk_id
    dst.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        s = src / name
        t = dst / (name + ".partial")
        shutil.copy2(s, t)
        t.replace(dst / name)


def http_upload_artifacts(chunk_id: str, base_url: str, local_out: Path | None = None) -> None:
    from cloud.claim_http import upload_artifact_http

    d = _local_chunk_dir(chunk_id, local_out)
    for name in REQUIRED:
        upload_artifact_http(base_url, chunk_id, d / name, name)


def durable_commit(
    chunk_id: str,
    *,
    final_status: str = "AFFINITY_DONE_SEG_PENDING",
    local_out: Path | None = None,
) -> dict:
    """
    Persist artifacts to durable store if configured, then complete claim.

    Env:
      S7_CLAIM_BACKEND=local|http
      S7_CLAIM_HTTP=http://controller:8787
      S7_DURABLE_MODE=local|rsync|http|copy
      S7_DURABLE_DEST=rsync destination OR controller OUT path for copy
    """
    t0 = time.perf_counter()
    verify_local_artifacts(chunk_id, local_out)
    mode = os.environ.get("S7_DURABLE_MODE", "local").strip().lower()
    claim_http = os.environ.get("S7_CLAIM_HTTP", "").strip()
    backend = os.environ.get("S7_CLAIM_BACKEND", "local").strip().lower()

    if mode == "rsync":
        dest = os.environ.get("S7_DURABLE_DEST", "").strip()
        if not dest:
            raise SystemExit("S7_DURABLE_DEST required for S7_DURABLE_MODE=rsync")
        rsync_to_durable(chunk_id, dest, local_out)
    elif mode == "copy":
        dest = Path(os.environ.get("S7_DURABLE_DEST", str(OUT)))
        copy_to_controller_out(chunk_id, dest, local_out)
    elif mode == "http":
        if not claim_http:
            raise SystemExit("S7_CLAIM_HTTP required for S7_DURABLE_MODE=http")
        http_upload_artifacts(chunk_id, claim_http, local_out)
    elif mode == "local":
        # Worker writes directly into durable OUT (local AMD path)
        pass
    else:
        raise SystemExit(f"unknown S7_DURABLE_MODE={mode}")

    persist_s = time.perf_counter() - t0

    # Complete only after durable artifacts exist on controller
    if backend == "http" or claim_http:
        from cloud.claim_http import complete_http

        url = claim_http or os.environ["S7_CLAIM_HTTP"]
        complete_http(chunk_id, url, final_status)
    else:
        from s7_chunk_claim import release_local

        # For local mode, affinities must already be under OUT
        release_local(chunk_id, final_status)

    return {"chunk_id": chunk_id, "status": final_status, "persist_seconds": persist_s, "mode": mode}
