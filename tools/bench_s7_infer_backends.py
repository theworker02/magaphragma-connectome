#!/usr/bin/env python3
"""A/B bench S7 MNet tile inference: eager vs torch.compile vs migraphx.

Uses synthetic (B,1,20,64,64) batches — same crop as production — so we do not
touch the claim queue. Stop any live FAST worker before running for a clean GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "third_party/segneuron/Train_and_Inference"))

CKPT = REPO / "experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt"
OUT = REPO / "experiments/phase6e/AFFINITY-S7-INFER-BACKEND-BENCH-001.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_model():
    import torch
    from model.Mnet import MNet
    from s7_infer_accelerate import accelerate_model

    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub")
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_weights"], strict=True)
    model = model.cuda().eval()
    return model, accelerate_model


def bench_one(model, batch: int, steps: int, warmup: int) -> dict:
    import torch

    x = torch.randn(batch, 1, 20, 64, 64, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    tiles = batch * steps
    return {
        "batch": batch,
        "steps": steps,
        "warmup": warmup,
        "tiles": tiles,
        "seconds": elapsed,
        "tiles_per_sec": tiles / elapsed if elapsed else None,
        "ms_per_tile": 1000.0 * elapsed / tiles if tiles else None,
        "vram_peak_bytes": int(torch.cuda.max_memory_allocated()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument(
        "--backends",
        default="eager,compile,migraphx",
        help="Comma list: eager,compile,migraphx,tensorrt",
    )
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm required")

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    results = []
    for name in backends:
        print(f"=== backend={name} ===", flush=True)
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
        base, accelerate_model = load_model()
        model, meta = accelerate_model(base, name, sample_batch=args.batch)
        print(json.dumps(meta), flush=True)
        try:
            tel = bench_one(model, args.batch, args.steps, args.warmup)
            tel["backend_meta"] = meta
            results.append({"backend": name, "ok": True, **tel})
            print(json.dumps(results[-1], indent=2), flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"backend": name, "ok": False, "error": str(exc), "backend_meta": meta})
            print(f"FAILED {name}: {exc}", flush=True)
        del model, base
        torch.cuda.empty_cache()

    ok = [r for r in results if r.get("ok") and r.get("tiles_per_sec")]
    winner = max(ok, key=lambda r: r["tiles_per_sec"]) if ok else None
    report = {
        "id": "AFFINITY_S7_INFER_BACKEND_BENCH_001",
        "created_at": _now(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "results": results,
        "winner": winner["backend"] if winner else None,
        "winner_tiles_per_sec": winner["tiles_per_sec"] if winner else None,
        "speedup_vs_eager": (
            (winner["tiles_per_sec"] / next(r["tiles_per_sec"] for r in ok if r["backend"] == "eager"))
            if winner and any(r["backend"] == "eager" for r in ok)
            else None
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("WROTE", OUT, flush=True)
    print(json.dumps({"winner": report["winner"], "speedup_vs_eager": report["speedup_vs_eager"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
