"""Mass-production benchmark, promotion gate, and receipts."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from hyperdrain import backends, pipeline
from hyperdrain.config import (
    BASELINE_COMPILE_TILES_PER_SEC,
    BASELINE_EAGER_TILES_PER_SEC,
    OUT,
    ensure_out,
)
from hyperdrain.equivalence import compare_volumes, performance_receipt
from hyperdrain.geometry import CROP_ZYX, STRIDE_ZYX
from hyperdrain.packed import infer_volume_packed

MASS_RECEIPT = OUT / "HYPERDRAIN_MASS_PRODUCTION_RECEIPT.json"
MASS_SPEEDUP = OUT / "HYPERDRAIN_MASS_PRODUCTION_SPEEDUP_VALIDATED.json"
PACK_CONFIG = OUT / "HYPERDRAIN_PACKED_CONFIG.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def production_tile_contract_hash() -> str:
    import hashlib

    payload = json.dumps(
        {
            "crop": list(CROP_ZYX),
            "stride": list(STRIDE_ZYX),
            "pad": "reflect",
            "norm": "/255",
            "stitch": "gaussian",
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def autotune_pack_size(
    model: Any,
    volume: np.ndarray,
    *,
    candidates: tuple[int, ...] | None = None,
    headroom_frac: float = 0.12,
) -> dict:
    """Find fastest stable pack size; OOM/regress stops search after a peak is found."""
    import os
    import torch

    ensure_out()
    if candidates is None:
        env = os.environ.get("HYPERDRAIN_PACK_CANDIDATES", "").strip()
        if env:
            candidates = tuple(int(x) for x in env.split(",") if x.strip())
        else:
            # Skip tiny packs — launch overhead dominates and falsely "wins" early.
            candidates = (4, 8, 16, 24, 32)
    warm_pack = 8 if 8 in candidates else min(candidates)
    infer_volume_packed(model, volume, pack_size=int(warm_pack), prefetch=False)
    torch.cuda.synchronize()

    trail: list[dict] = []
    best: dict[str, Any] = {"pack_size": int(warm_pack), "wall_seconds": float("inf"), "tiles_per_second": 0.0}
    prev_tps = 0.0
    for pack in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, _, tel = infer_volume_packed(model, volume, pack_size=int(pack), prefetch=True)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            peak = int(torch.cuda.max_memory_allocated())
            total = torch.cuda.get_device_properties(0).total_memory
            budget = int(total * (1.0 - headroom_frac))
            tps = float(tel.get("tiles_per_second") or 0.0)
            row = {
                "pack_size": pack,
                "wall_seconds": wall,
                "tiles_per_second": tps,
                "vram_peak_bytes": peak,
                "ok": peak <= budget,
            }
            trail.append(row)
            print(
                f"  autotune pack={pack} wall={wall:.2f}s tps={tps:.2f} peak_gb={peak / 1e9:.2f}",
                flush=True,
            )
            if peak > budget:
                print(f"  autotune: pack={pack} exceeds headroom — stop", flush=True)
                break
            if wall < float(best["wall_seconds"]):
                best = {
                    "pack_size": pack,
                    "wall_seconds": wall,
                    "tiles_per_second": tps,
                    "vram_peak_bytes": peak,
                }
            if prev_tps > 0 and tps < prev_tps * 0.92 and int(best["pack_size"]) >= 8 and pack > int(best["pack_size"]):
                print(f"  autotune: throughput regress at pack={pack} — stop", flush=True)
                break
            prev_tps = tps
        except Exception as exc:  # noqa: BLE001
            trail.append({"pack_size": pack, "ok": False, "error": str(exc)[:200]})
            print(f"  autotune pack={pack} FAIL {exc} — stop", flush=True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            break

    cfg = {
        "id": "HYPERDRAIN_PACKED_CONFIG",
        "created_at": _now(),
        "pack_size": int(best["pack_size"]),
        "prefetch": True,
        "buffers": 2,
        "backend": "eager",
        "trail": trail,
        "best": best,
        "production_tile_contract_hash": production_tile_contract_hash(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    PACK_CONFIG.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cfg


def load_packed_config() -> dict:
    if PACK_CONFIG.exists():
        return json.loads(PACK_CONFIG.read_text(encoding="utf-8"))
    return {"pack_size": 16, "prefetch": True, "buffers": 2, "backend": "eager"}


def run_mass_benchmark(
    model_eager: Any,
    volume: np.ndarray,
    *,
    also_compile: bool = False,
    model_compile: Any | None = None,
) -> dict:
    """Warm end-to-end wall-clock: eager baseline vs packed (+ optional compile)."""
    import torch

    ensure_out()
    contract = production_tile_contract_hash()

    print("mass-benchmark: warm discard (eager tiled)...", flush=True)
    pipeline.infer_volume_tiled(model_eager, volume, batch_size=16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ref_aff, ref_bnd, ref_tel = pipeline.infer_volume_tiled(model_eager, volume, batch_size=16)
    torch.cuda.synchronize()
    ref_wall = time.perf_counter() - t0
    ref_tel = {**ref_tel, "wall_seconds": ref_wall, "role": "reference_eager_tiled"}

    print("mass-benchmark: autotune pack size...", flush=True)
    pack_cfg = autotune_pack_size(model_eager, volume)
    pack = int(pack_cfg["pack_size"])

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    p_aff, p_bnd, p_tel = infer_volume_packed(model_eager, volume, pack_size=pack, prefetch=True)
    torch.cuda.synchronize()
    p_wall = time.perf_counter() - t1
    p_tel = {**p_tel, "wall_seconds": p_wall, "role": "packed_eager"}

    cmp = compare_volumes(ref_aff, p_aff, ref_bnd, p_bnd)
    speedup = (ref_wall / p_wall) if p_wall > 0 else None
    wall_win = bool(
        cmp.get("pass")
        and p_wall < ref_wall
        and ref_wall > 0
        and pack >= 4
        and speedup is not None
        and speedup >= 1.02
    )

    compile_block = None
    if also_compile and model_compile is not None:
        try:
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            c_aff, c_bnd, c_tel = infer_volume_packed(model_compile, volume, pack_size=pack, prefetch=True)
            torch.cuda.synchronize()
            c_wall = time.perf_counter() - t2
            c_cmp = compare_volumes(ref_aff, c_aff, ref_bnd, c_bnd)
            compile_block = {
                "wall_seconds": c_wall,
                "tiles_per_second": c_tel.get("tiles_per_second"),
                "equivalence": c_cmp,
                "rejected": not bool(c_cmp.get("pass")),
            }
            if not c_cmp.get("pass"):
                print("mass-benchmark: packed+compile REJECTED (equivalence)", flush=True)
        except Exception as exc:  # noqa: BLE001
            compile_block = {"rejected": True, "error": str(exc)[:300]}

    declare = wall_win
    receipt = {
        "id": "HYPERDRAIN_MASS_PRODUCTION_RECEIPT",
        "created_at": _now(),
        "production_tile_contract_hash": contract,
        "volume_zyx": list(volume.shape),
        "reference": {
            "backend": "eager",
            "mode": "tiled",
            "tiles_per_second": ref_tel.get("tiles_per_second"),
            "chunk_wall_seconds": ref_wall,
            "preserved_baseline_eager_tiles_per_sec": BASELINE_EAGER_TILES_PER_SEC,
            "preserved_baseline_compile_tiles_per_sec": BASELINE_COMPILE_TILES_PER_SEC,
        },
        "selected": {
            "backend": "packed-eager",
            "pack_size": pack,
            "prefetch_depth": 1,
            "buffers": 2,
            "tiles_per_second": p_tel.get("tiles_per_second"),
            "chunk_wall_seconds": p_wall,
        },
        "equivalence": cmp,
        "speedup": {
            "per_GPU_wall_clock_factor": speedup if cmp.get("pass") else None,
            "per_GPU_throughput_factor": (
                (p_tel.get("tiles_per_second") or 0) / max(1e-9, ref_tel.get("tiles_per_second") or 1e-9)
                if cmp.get("pass")
                else None
            ),
            "MASS_PRODUCTION_SPEEDUP_VALIDATED": declare,
        },
        "compile_probe": compile_block,
        "pack_autotune": pack_cfg,
        "fleet": {
            "active_GPUs": 1,
            "aggregate_tiles_per_second": p_tel.get("tiles_per_second") if declare else ref_tel.get("tiles_per_second"),
            "note": "Fleet aggregates filled by status from live workers",
        },
        "recovery": {"retry_count": 0, "resumed_chunks": 0, "avoided_recomputed_tiles": 0},
        "verdict": (
            "HYPERDRAIN_MASS_PRODUCTION_SPEEDUP_VALIDATED"
            if declare
            else ("EQUIVALENCE_FAIL" if not cmp.get("pass") else "NO_WALL_IMPROVEMENT")
        ),
    }
    MASS_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    q = backends.load_qualified()
    q["decisive_metric"] = "verified_production_chunks_per_wall_clock_hour"
    q["mass_production_speedup_validated"] = declare
    notes = dict(q.get("notes") or {})
    if declare:
        MASS_SPEEDUP.write_text(
            json.dumps(
                {
                    "id": "HYPERDRAIN_MASS_PRODUCTION_SPEEDUP_VALIDATED",
                    "declared_at": _now(),
                    "backend": "packed-eager",
                    "pack_size": pack,
                    "reference_wall_seconds": ref_wall,
                    "selected_wall_seconds": p_wall,
                    "wall_speedup_factor": speedup,
                    "equivalence_pass": True,
                    "production_tile_contract_hash": contract,
                    "evidence": str(MASS_RECEIPT),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        q.setdefault("qualified", [])
        if "packed" not in q["qualified"]:
            q["qualified"].append("packed")
        q["production"] = "packed"
        notes["packed"] = (
            f"MASS_PRODUCTION_SPEEDUP_VALIDATED: packed-eager pack={pack} "
            f"wall {ref_wall:.3f}s -> {p_wall:.3f}s ({speedup:.3f}x) on {list(volume.shape)}"
        )
    else:
        if MASS_SPEEDUP.exists():
            MASS_SPEEDUP.unlink()
        notes["packed"] = (
            f"Not promoted. verdict={receipt['verdict']} "
            f"ref_wall={ref_wall:.3f} packed_wall={p_wall:.3f} equiv={cmp.get('pass')}"
        )
        experimental = list(q.get("experimental") or [])
        if "packed" not in experimental:
            experimental.append("packed")
        q["experimental"] = experimental
    q["notes"] = notes
    backends.save_qualified(q)

    performance_receipt(
        measurements=[
            {
                "role": "reference_eager",
                "wall_seconds": ref_wall,
                "tiles_per_second": ref_tel.get("tiles_per_second"),
            },
            {
                "role": "packed_eager",
                "wall_seconds": p_wall,
                "tiles_per_second": p_tel.get("tiles_per_second"),
                "pack_size": pack,
            },
        ],
        notes=f"mass-benchmark verdict={receipt['verdict']}",
    )
    return receipt