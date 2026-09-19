"""Fleet scaling — the path to 5x / 25x / 30x wall-clock when per-GPU is at ceiling."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperdrain.config import BASELINE_EAGER_TILES_PER_SEC, OUT, ensure_out
from hyperdrain import queue

FLEET_PLAN = OUT / "HYPERDRAIN_FLEET_PLAN.json"
TILES_PER_CHUNK_FAST = 3468  # measured FAST core tiles, not the obsolete 20736 hint


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def measured_tiles_per_sec_per_gpu() -> float:
    cfg = OUT / "HYPERDRAIN_PACKED_CONFIG.json"
    if cfg.exists():
        try:
            best = json.loads(cfg.read_text(encoding="utf-8")).get("best") or {}
            tps = float(best.get("tiles_per_second") or 0.0)
            if tps > 0:
                return tps
        except Exception:
            pass
    return float(BASELINE_EAGER_TILES_PER_SEC)


def project_fleet(n_gpus: int, tiles_per_sec_per_gpu: float | None = None) -> dict[str, Any]:
    """
    Calendar speedup from N independent packed workers.

    Per-GPU MNet forward ceiling on RX 7800 XT is ~19 tiles/s. Packing cannot
    invent a 25x kernel win. Fleet throughput scales ~linearly:
        T_wall ≈ T_single / N_effective_GPUs
    """
    ensure_out()
    n = max(1, int(n_gpus))
    tps1 = float(tiles_per_sec_per_gpu or measured_tiles_per_sec_per_gpu())
    summary = queue.status_summary()
    remaining = max(0, summary["n_total"] - summary["n_done"])
    agg_tps = tps1 * n
    cph = (agg_tps * 3600.0) / TILES_PER_CHUNK_FAST
    cph1 = (tps1 * 3600.0) / TILES_PER_CHUNK_FAST
    eta_h = remaining / cph if cph > 0 else None
    eta1_h = remaining / cph1 if cph1 > 0 else None
    plan = {
        "id": "HYPERDRAIN_FLEET_PLAN",
        "created_at": _now(),
        "n_gpus": n,
        "tiles_per_sec_per_gpu": tps1,
        "aggregate_tiles_per_sec": agg_tps,
        "tiles_per_chunk": TILES_PER_CHUNK_FAST,
        "chunks_per_hour_per_gpu": cph1,
        "aggregate_chunks_per_hour": cph,
        "chunks_remaining": remaining,
        "eta_hours_one_gpu": eta1_h,
        "eta_hours_fleet": eta_h,
        "wall_speedup_vs_one_gpu": float(n),
        "per_gpu_ceiling_note": (
            "Pure MNet forward on RX 7800 XT measures ~19 tiles/s. "
            "Packed production sits at that ceiling (~18 tiles/s). "
            "25–30× calendar time requires ~25–30 GPUs, not a 25× single-GPU kernel."
        ),
        "formula": "T_wall ≈ T_single_GPU / N_effective_GPUs",
    }
    FLEET_PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def launch_local_workers(n: int, max_chunks: int = 64) -> dict[str, Any]:
    """
    Spawn N packed production worker processes on this machine.

    Note: one discrete GPU cannot run N full workers usefully — they contend.
    Use n=1 locally; use this launcher on each cloud GPU VM (n=1 per VM).
    """
    ensure_out()
    repo = Path(__file__).resolve().parents[2]
    tools = repo / "tools"
    py = os.environ.get("HYPERDRAIN_PYTHON") or sys.executable
    log_dir = OUT / "fleet_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(max(1, int(n))):
        wid = f"fleet-{i}-{os.getpid()}"
        log = log_dir / f"{wid}.log"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(tools) + os.pathsep + env.get("PYTHONPATH", "")
        env["HYPERDRAIN_WORKER_ID"] = wid
        env["HYPERDRAIN_SKIP_VRAM_SEARCH"] = env.get("HYPERDRAIN_SKIP_VRAM_SEARCH", "1")
        cmd = [
            py,
            "-u",
            str(tools / "hyperdrain.py"),
            "production",
            "--max-chunks",
            str(max_chunks),
        ]
        fh = open(log, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(repo), env=env, stdout=fh, stderr=subprocess.STDOUT)
        procs.append({"worker_id": wid, "pid": proc.pid, "log": str(log)})
    return {"launched": procs, "warning": "Multiple workers on one GPU contend — prefer one worker per GPU host."}


def format_fleet_report(n_gpus: int) -> str:
    p = project_fleet(n_gpus)
    eta = p.get("eta_hours_fleet")
    eta1 = p.get("eta_hours_one_gpu")

    def fmt(h: float | None) -> str:
        if h is None:
            return "n/a"
        d, r = divmod(float(h), 24.0)
        return f"{int(d)}d {int(r)}h"

    return (
        "HYPERDRAIN FLEET PROJECTION\n\n"
        f"  GPUs requested:     {p['n_gpus']}\n"
        f"  tiles/s / GPU:      {p['tiles_per_sec_per_gpu']:.1f}\n"
        f"  aggregate tiles/s:  {p['aggregate_tiles_per_sec']:.1f}\n"
        f"  chunks/hour:        {p['aggregate_chunks_per_hour']:.1f}\n"
        f"  wall speedup:       {p['wall_speedup_vs_one_gpu']:.0f}× vs 1 GPU\n"
        f"  ETA (1 GPU):        {fmt(eta1)}\n"
        f"  ETA ({p['n_gpus']} GPUs):     {fmt(eta)}\n\n"
        f"  {p['per_gpu_ceiling_note']}\n"
    )