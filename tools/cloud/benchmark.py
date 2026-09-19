"""Local + remote Affinity qualification benchmarks (equivalence-gated)."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# tools/ is on sys.path when imported from affinity_vast.py
REPO = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO / "experiments/phase6e/VAST-AFFINITY-BENCH-001"
TILES_PER_CHUNK = 3468
BATCH_SWEEP = [8, 16, 24, 32, 48, 64]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def remaining_chunks() -> int:
    prog = REPO / "experiments/phase6e/AFFINITY-FULLVOL-S7-001/PROGRESS.json"
    if prog.exists():
        data = json.loads(prog.read_text(encoding="utf-8"))
        return int(data.get("n_not_started", data.get("n_total", 28798) - data.get("n_done", 0)))
    return 28598


def run_local_batch_qualification(
    *,
    batches: list[int] | None = None,
    backend: str = "eager",
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run production tile-math batch sweep on the local GPU with equivalence gates.
    Does not rent Vast instances. Records receipts under VAST-AFFINITY-BENCH-001.
    """
    import numpy as np
    import torch

    import sys

    sys.path.insert(0, str(REPO / "tools"))
    from hyperdrain.equivalence import compare_volumes
    from hyperdrain.geometry import CROP_ZYX, STRIDE_ZYX, tile_layout
    from hyperdrain.worker import load_model
    from run_affinity_fullvol_s7_fast_worker import infer_volume_batched

    out_dir = out_dir or BENCH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = batches or BATCH_SWEEP

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm required for local qualification")

    device_name = torch.cuda.get_device_name(0)
    # Production-shaped but smaller volume for wall-clock; tile math identical.
    shape = (64, 256, 256)
    n_tiles = tile_layout(shape, CROP_ZYX, STRIDE_ZYX).n_tiles
    rng = np.random.default_rng(20260918)
    volume = rng.integers(0, 256, size=shape, dtype=np.uint8)

    model, meta, cfg = load_model(backend if backend != "compile" else "eager")
    model.eval()
    torch.backends.cudnn.benchmark = True
    if backend == "compile":
        from s7_infer_accelerate import accelerate_model

        model, backend_meta = accelerate_model(model, "compile", sample_batch=16)
    else:
        backend_meta = {"applied": "eager", "requested": backend}

    ref_aff = ref_bnd = None
    runs = []
    for batch in batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.perf_counter()
            aff, bnd, tel = infer_volume_batched(model, volume, device="cuda", batch_size=batch)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            finite = bool(np.isfinite(aff).all() and np.isfinite(bnd).all())
            if ref_aff is None:
                ref_aff, ref_bnd = aff, bnd
                equiv = {"pass": True, "reason": "reference"}
            else:
                equiv = compare_volumes(ref_aff, aff, ref_bnd, bnd)
            tps = n_tiles / wall if wall > 0 else 0.0
            # Scale note: TPS measured on probe volume; production path identical per tile.
            row = {
                "timestamp": _now(),
                "device": device_name,
                "backend": backend_meta.get("applied", backend),
                "batch": batch,
                "probe_shape_zyx": list(shape),
                "probe_tiles": n_tiles,
                "tiles_per_second": tps,
                "seconds_per_chunk_est": TILES_PER_CHUNK / tps if tps > 0 else None,
                "chunks_per_hour_est": (tps / TILES_PER_CHUNK) * 3600.0 if tps > 0 else None,
                "vram_peak_bytes": int(torch.cuda.max_memory_allocated()),
                "finite": finite,
                "equivalence": equiv,
                "crop_zyx": list(CROP_ZYX),
                "stride_zyx": list(STRIDE_ZYX),
            }
            runs.append(row)
            if not finite or not equiv.get("pass"):
                break
            if len(runs) >= 3:
                tps_list = [r["tiles_per_second"] for r in runs]
                if tps_list[-1] < tps_list[-2] < tps_list[-3]:
                    break
        except RuntimeError as e:
            runs.append({"batch": batch, "error": str(e), "timestamp": _now()})
            break

    ok = [r for r in runs if r.get("tiles_per_second") and r.get("finite") and r.get("equivalence", {}).get("pass")]
    best = max(ok, key=lambda r: r["tiles_per_second"]) if ok else None
    report = {
        "id": "VAST_LOCAL_BATCH_QUALIFICATION",
        "created_at": _now(),
        "torch": torch.__version__,
        "backend_meta": backend_meta,
        "runs": runs,
        "best": best,
        "remaining_chunks": remaining_chunks(),
    }
    path = out_dir / "VAST_GPU_BENCHMARKS.json"
    # merge if exists
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        prev.setdefault("local_qualifications", []).append(report)
        path.write_text(json.dumps(prev, indent=2) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps({"local_qualifications": [report]}, indent=2) + "\n", encoding="utf-8")
    return report


def write_cost_model_receipt(
    *,
    tiles_per_second: float,
    gpu_hourly_usd: float,
    gpu_name: str,
    offer_id: int | None = None,
    out_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .pricing import estimate_chunk_economics

    out_dir = out_dir or BENCH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rem = remaining_chunks()
    model = estimate_chunk_economics(
        tiles_per_second=tiles_per_second,
        gpu_hourly_usd=gpu_hourly_usd,
        remaining_chunks=rem,
    )
    receipt = {
        "id": "VAST_COST_MODEL",
        "created_at": _now(),
        "gpu_name": gpu_name,
        "offer_id": offer_id,
        "model": model.to_dict(),
        **(extra or {}),
    }
    (out_dir / "VAST_COST_MODEL.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
