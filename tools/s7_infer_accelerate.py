"""Optional inference acceleration backends for S7 MNet (eager / torch.compile / MIGraphX)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _ensure_rocm_migraphx_pythonpath() -> list[str]:
    """ROCm ships migraphx.cpython-*.so under /opt/rocm*/lib; torch_migraphx needs that on PYTHONPATH."""
    added: list[str] = []
    candidates = []
    rocm = os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH")
    if rocm:
        candidates.append(Path(rocm) / "lib")
    candidates.extend(sorted(Path("/opt").glob("rocm*/lib"), reverse=True))
    for lib in candidates:
        so = next(lib.glob("migraphx.cpython-*.so"), None)
        if so is None:
            continue
        text = str(lib)
        if text not in sys.path:
            sys.path.insert(0, text)
            added.append(text)
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        extras = [text, str(lib / "migraphx" / "lib")]
        for extra in extras:
            if extra and Path(extra).exists() and extra not in ld.split(":"):
                ld = f"{extra}:{ld}" if ld else extra
        os.environ["LD_LIBRARY_PATH"] = ld
        break
    return added


def resolve_backend(cli: str | None = None) -> str:
    raw = (cli or os.environ.get("S7_INFER_BACKEND") or "eager").strip().lower()
    if raw in {"eager", "compile", "migraphx", "tensorrt"}:
        return raw
    raise ValueError(f"unknown infer backend: {raw}")


def accelerate_model(model: Any, backend: str, sample_batch: int = 16) -> tuple[Any, dict]:
    """Return (model_or_compiled, meta). Never silently falls back without recording it."""
    import torch

    backend = resolve_backend(backend)
    meta: dict = {"requested": backend, "applied": "eager", "detail": None, "migraphx_path_added": []}
    model = model.eval()
    if backend == "eager":
        return model, meta

    sample = torch.randn(sample_batch, 1, 20, 64, 64, device="cuda", dtype=torch.float32)

    if backend == "compile":
        compiled = torch.compile(model, mode="default", fullgraph=False)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = compiled(sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        meta["applied"] = "compile"
        meta["detail"] = "torch.compile(mode=default)"
        return compiled, meta

    if backend == "migraphx":
        meta["migraphx_path_added"] = _ensure_rocm_migraphx_pythonpath()
        try:
            import migraphx  # noqa: F401
            import torch_migraphx  # noqa: F401
        except ImportError as exc:
            meta["detail"] = f"torch_migraphx/migraphx import failed: {exc}"
            return model, meta
        try:
            compiled = torch.compile(model, backend="migraphx")
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = compiled(sample)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            meta["applied"] = "migraphx"
            meta["detail"] = "torch.compile(backend=migraphx)"
            return compiled, meta
        except Exception as exc:  # noqa: BLE001
            meta["detail"] = f"migraphx compile failed: {exc}"
            return model, meta

    if backend == "tensorrt":
        try:
            import torch_tensorrt  # noqa: F401
        except ImportError as exc:
            meta["detail"] = f"torch_tensorrt import failed: {exc}"
            return model, meta
        try:
            compiled = torch.compile(model, backend="tensorrt")
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = compiled(sample)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            meta["applied"] = "tensorrt"
            meta["detail"] = "torch.compile(backend=tensorrt)"
            return compiled, meta
        except Exception as exc:  # noqa: BLE001
            meta["detail"] = f"tensorrt compile failed: {exc}"
            return model, meta

    return model, meta
