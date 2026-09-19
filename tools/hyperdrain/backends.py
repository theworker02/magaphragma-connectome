"""Execution backends: auto | eager | compile | migraphx (experimental)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperdrain.config import QUALIFIED_MANIFEST, ensure_out


def normalize_backend(name: str) -> str:
    raw = name.strip().lower()
    if raw.startswith("hyperdrain:"):
        raw = raw.split(":", 1)[1]
    if raw in {"auto", "eager", "compile", "migraphx", "dense", "fast", "packed"}:
        if raw == "fast":
            return "compile"
        if raw == "packed":
            return "packed"
        return raw
    raise ValueError(f"unknown backend: {name}")


def load_qualified() -> dict:
    if not QUALIFIED_MANIFEST.exists():
        return {"qualified": ["eager"], "production": "eager", "experimental": ["migraphx"]}
    return json.loads(QUALIFIED_MANIFEST.read_text(encoding="utf-8"))


def save_qualified(data: dict) -> None:
    ensure_out()
    tmp = QUALIFIED_MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(QUALIFIED_MANIFEST)


def resolve_auto() -> str:
    """Pick fastest previously qualified backend; never select unqualified migraphx."""
    q = load_qualified()
    prod = q.get("production")
    allowed = set(q.get("qualified") or ["eager"])
    if prod in allowed:
        return prod
    # Prefer compile if qualified else eager
    for cand in ("packed", "compile", "eager"):
        if cand in allowed:
            return cand
    return "eager"


def accelerate(model: Any, backend: str, sample_shape: tuple[int, ...] = (16, 1, 20, 64, 64)) -> tuple[Any, dict]:
    """
    Wrap model. Reuses tools.s7_infer_accelerate when available.
    MIGraphX failures fall back to eager and record detail — never crash the queue.
    """
    import torch

    backend = normalize_backend(backend)
    if backend == "auto":
        backend = resolve_auto()
    if backend == "dense":
        # Dense is a pipeline mode; underlying module stays eager/compile via env later.
        backend = "eager"
    requested = backend
    if backend == "packed":
        # Packed is a scheduling mode over production tiles; module stays eager unless compile selected.
        backend = "eager"

    meta: dict = {"requested": requested, "applied": "eager", "detail": None}
    if requested == "packed":
        meta["applied"] = "packed-eager"
        meta["detail"] = "production tile packing (immutable tile math)"

    # Prefer existing helper for compile/migraphx parity with S7.
    try:
        import sys
        from pathlib import Path

        tools = Path(__file__).resolve().parents[1]
        if str(tools) not in sys.path:
            sys.path.insert(0, str(tools))
        from s7_infer_accelerate import accelerate_model

        if backend in {"eager", "compile", "migraphx"}:
            return accelerate_model(model, backend, sample_batch=int(sample_shape[0]))
    except Exception as exc:  # noqa: BLE001
        meta["detail"] = f"s7_infer_accelerate unavailable: {exc}"

    model = model.eval()
    if backend == "eager":
        return model, meta

    if backend == "compile":
        try:
            sample = torch.randn(*sample_shape, device="cuda", dtype=torch.float32)
            compiled = torch.compile(model, mode="default", fullgraph=False)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = compiled(sample)
            torch.cuda.synchronize()
            meta["applied"] = "compile"
            meta["detail"] = "torch.compile(mode=default)"
            return compiled, meta
        except Exception as exc:  # noqa: BLE001
            meta["detail"] = f"compile failed: {exc}"
            return model, meta

    if backend == "migraphx":
        meta["detail"] = "migraphx experimental — use isolated worker; falling back to eager"
        return model, meta

    return model, meta
