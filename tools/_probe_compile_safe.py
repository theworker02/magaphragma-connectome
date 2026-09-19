import os, sys, time, json
from pathlib import Path

ROOT = Path("/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome")
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "third_party/segneuron/Train_and_Inference"))
os.environ["HYPERDRAIN_SKIP_VRAM_SEARCH"] = "1"
os.chdir(ROOT)

import torch
import numpy as np
from hyperdrain.worker import load_model


def bench(m, batch=8, iters=35, warmup=12):
    x = torch.randn(batch, 1, 20, 64, 64, device="cuda")
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast("cuda", torch.float16):
                a, b = m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.autocast("cuda", torch.float16):
                a, b = m(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return {
        "tps": (iters * batch) / dt,
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


results = []
m = load_model("eager")[0]
torch.backends.cudnn.benchmark = True
r = bench(m)
r["label"] = "baseline"
results.append(r)
print(r, flush=True)

import model.Mnet as MM

_orig = MM.FMU


def FMU_fp32(x1, x2, mode="sub"):
    if mode == "sum":
        return torch.add(x1, x2)
    if mode == "sub":
        return torch.abs(x1.float() - x2.float())
    if mode == "cat":
        return torch.cat((x1, x2), dim=1)
    raise Exception(mode)


MM.FMU = FMU_fp32
print("compiling...", flush=True)
compiled = torch.compile(m, mode="reduce-overhead", fullgraph=False)
r = bench(compiled, iters=25, warmup=25)
r["label"] = "compile_fmu_fp32"
results.append(r)
print(r, flush=True)

if r["finite"]:
    from hyperdrain import pipeline
    from hyperdrain.equivalence import compare_volumes

    MM.FMU = _orig
    ref = load_model("eager")[0]
    MM.FMU = FMU_fp32
    vol = np.random.default_rng(0).integers(0, 256, size=(48, 192, 192), dtype=np.uint8)
    pipeline.infer_volume_tiled(ref, vol, batch_size=8)
    pipeline.infer_volume_tiled(compiled, vol, batch_size=8)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ra, rb, _ = pipeline.infer_volume_tiled(ref, vol, batch_size=8)
    torch.cuda.synchronize()
    w0 = time.perf_counter() - t0
    t1 = time.perf_counter()
    ca, cb, _ = pipeline.infer_volume_tiled(compiled, vol, batch_size=8)
    torch.cuda.synchronize()
    w1 = time.perf_counter() - t1
    cmp = compare_volumes(ra, ca, rb, cb)
    row = {
        "label": "equiv",
        "pass": cmp.get("pass"),
        "stats": cmp.get("stats"),
        "ref_wall": w0,
        "cand_wall": w1,
        "speedup": (w0 / w1) if w1 else None,
    }
    results.append(row)
    print(row, flush=True)

path = ROOT / "experiments/phase6e/HYPERDRAIN/SINGLE_GPU_SPEED_PROBE2.json"
path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print("WROTE", path, flush=True)
