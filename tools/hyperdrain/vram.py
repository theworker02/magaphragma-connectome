"""Adaptive VRAM optimizer with cache + OOM binary search."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from hyperdrain.config import VRAM_CACHE, ensure_out
from hyperdrain.geometry import CROP_ZYX


@dataclass
class VramConfig:
    batch_size: int
    macro_zyx: tuple[int, int, int]
    backend: str
    precision: str
    tiles_per_sec: float
    vram_peak_bytes: int
    headroom_frac: float
    gpu_name: str
    torch_version: str
    hip_version: str | None
    model_hash: str
    stable: bool


def model_weights_hash(path: Path) -> str:
    import os
    if os.environ.get("HYPERDRAIN_FULL_CKPT_HASH") == "1":
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    st = path.stat()
    return hashlib.sha256(f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]


def cache_key(cfg_id: dict) -> str:
    return hashlib.sha256(json.dumps(cfg_id, sort_keys=True).encode()).hexdigest()[:24]


def load_cache() -> dict:
    if not VRAM_CACHE.exists():
        return {}
    return json.loads(VRAM_CACHE.read_text(encoding="utf-8"))


def save_cache(cache: dict) -> None:
    ensure_out()
    tmp = VRAM_CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(VRAM_CACHE)


def _oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "hiperroroutofmemory" in msg or "cuda out of memory" in msg


def probe_forward(
    model: Any,
    shape: tuple[int, ...],
    precision: str = "fp16",
) -> tuple[float, int]:
    """Return (seconds_per_forward, peak_bytes) for one warmup+timed forward."""
    import torch

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    amp = precision in {"fp16", "bf16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    # warmup
    with torch.inference_mode():
        for _ in range(2):
            if amp:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _ = model(x)
            else:
                _ = model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if amp:
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = model(x)
        else:
            _ = model(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    peak = int(torch.cuda.max_memory_allocated())
    del x
    return dt, peak


def binary_search_batch(
    model: Any,
    zyx: tuple[int, int, int] = CROP_ZYX,
    precision: str = "fp16",
    lo: int = 1,
    hi: int = 64,
    headroom_frac: float = 0.12,
) -> tuple[int, dict]:
    """Largest batch that fits with headroom; records OOM trail."""
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    budget = int(total * (1.0 - headroom_frac))
    trail = []
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            dt, peak = probe_forward(model, (mid, 1, *zyx), precision=precision)
            trail.append({"batch": mid, "ok": True, "peak": peak, "dt": dt})
            if peak <= budget:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        except Exception as exc:  # noqa: BLE001
            trail.append({"batch": mid, "ok": False, "error": str(exc), "oom": _oom(exc)})
            torch.cuda.empty_cache()
            hi = mid - 1
    return best, {"trail": trail, "budget_bytes": budget, "total_bytes": total}


def binary_search_macro_z(
    model: Any,
    y: int = 64,
    x: int = 64,
    precision: str = "fp16",
    z_lo: int = 20,
    z_hi: int = 128,
    batch: int = 1,
    headroom_frac: float = 0.12,
) -> tuple[tuple[int, int, int], dict]:
    """Largest Z for macro tile at fixed YX."""
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    budget = int(total * (1.0 - headroom_frac))
    trail = []
    best_z = z_lo
    lo, hi = z_lo, z_hi
    while lo <= hi:
        mid = (lo + hi) // 2
        # keep even-ish for pooling
        mid = mid - (mid % 2)
        mid = max(z_lo, mid)
        try:
            dt, peak = probe_forward(model, (batch, 1, mid, y, x), precision=precision)
            trail.append({"zyx": [mid, y, x], "ok": True, "peak": peak, "dt": dt})
            if peak <= budget:
                best_z = mid
                lo = mid + 2
            else:
                hi = mid - 2
        except Exception as exc:  # noqa: BLE001
            trail.append({"zyx": [mid, y, x], "ok": False, "error": str(exc), "oom": _oom(exc)})
            torch.cuda.empty_cache()
            hi = mid - 2
    return (best_z, y, x), {"trail": trail, "budget_bytes": budget}


def optimize(
    model: Any,
    backend: str,
    ckpt_path: Path,
    precision: str = "fp16",
    force: bool = False,
) -> VramConfig:
    import torch

    ensure_out()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    ident = {
        "gpu": gpu,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "model_hash": model_weights_hash(ckpt_path),
        "backend": backend,
        "precision": precision,
        "crop": list(CROP_ZYX),
    }
    key = cache_key(ident)
    cache = load_cache()
    if not force and key in cache and cache[key].get("stable"):
        c = cache[key]
        return VramConfig(
            batch_size=int(c["batch_size"]),
            macro_zyx=tuple(c["macro_zyx"]),  # type: ignore[arg-type]
            backend=c["backend"],
            precision=c["precision"],
            tiles_per_sec=float(c["tiles_per_sec"]),
            vram_peak_bytes=int(c["vram_peak_bytes"]),
            headroom_frac=float(c["headroom_frac"]),
            gpu_name=c["gpu_name"],
            torch_version=c["torch_version"],
            hip_version=c.get("hip_version"),
            model_hash=c["model_hash"],
            stable=True,
        )

    print("hyperdrain vram: probing batch 8..24...", flush=True)
    best_batch, batch_meta = binary_search_batch(model, precision=precision, lo=8, hi=24)
    print(f"hyperdrain vram: best_batch={best_batch}", flush=True)
    print("hyperdrain vram: probing macro Z 20..64...", flush=True)
    macro, macro_meta = binary_search_macro_z(model, precision=precision, batch=1, z_lo=20, z_hi=64)
    print(f"hyperdrain vram: macro={macro}", flush=True)
    dt, peak = probe_forward(model, (best_batch, 1, *CROP_ZYX), precision=precision)
    tps = (best_batch / dt) if dt > 0 else 0.0

    cfg = VramConfig(
        batch_size=best_batch,
        macro_zyx=macro,
        backend=backend,
        precision=precision,
        tiles_per_sec=tps,
        vram_peak_bytes=peak,
        headroom_frac=0.12,
        gpu_name=gpu,
        torch_version=torch.__version__,
        hip_version=getattr(torch.version, "hip", None),
        model_hash=ident["model_hash"],
        stable=True,
    )
    cache[key] = {**asdict(cfg), "macro_zyx": list(cfg.macro_zyx), "batch_meta": batch_meta, "macro_meta": macro_meta}
    save_cache(cache)
    return cfg
