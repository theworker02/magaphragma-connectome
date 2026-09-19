import os, sys, time, json
from pathlib import Path

ROOT = Path("/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome")
sys.path.insert(0, str(ROOT / "tools"))
os.environ["HYPERDRAIN_SKIP_VRAM_SEARCH"] = "1"
os.chdir(ROOT)

import torch
import numpy as np

print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None), flush=True)
print("gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)
for m in ("onnx", "onnxruntime", "torch_migraphx", "migraphx"):
    try:
        __import__(m)
        print(m, "YES", flush=True)
    except Exception as e:
        print(m, "NO", type(e).__name__, flush=True)

from hyperdrain.worker import load_model
from hyperdrain.packed import infer_volume_packed
from hyperdrain import pipeline
from hyperdrain.equivalence import compare_volumes
from hyperdrain.geometry import CROP_ZYX, STRIDE_ZYX

model, meta, cfg = load_model("eager")
model.eval()
torch.backends.cudnn.benchmark = True

# Count params / look at modules
nparams = sum(p.numel() for p in model.parameters())
print(f"params={nparams/1e6:.2f}M", flush=True)

def bench_raw(m, batch=8, iters=40, warmup=15, use_amp=True, channels_last=False):
    x = torch.randn(batch, 1, 20, 64, 64, device="cuda", dtype=torch.float32)
    if channels_last:
        try:
            x = x.to(memory_format=torch.channels_last_3d)
            m2 = m.to(memory_format=torch.channels_last_3d)
        except Exception as e:
            print("channels_last_3d unsupported", e, flush=True)
            m2 = m
    else:
        m2 = m
    with torch.inference_mode():
        for _ in range(warmup):
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    _ = m2(x)
            else:
                _ = m2(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = m2(x)
            else:
                out = m2(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    # nan check last out
    if isinstance(out, (tuple, list)):
        finite = all(torch.isfinite(o).all().item() for o in out)
    else:
        finite = torch.isfinite(out).all().item()
    tps = (iters * batch) / dt
    return {"tps": tps, "ms_tile": 1000 * dt / (iters * batch), "finite": finite}

results = []
print("=== eager variants ===", flush=True)
for label, kwargs in [
    ("eager_amp_b8", dict(batch=8, use_amp=True)),
    ("eager_amp_b16", dict(batch=16, use_amp=True)),
    ("eager_fp32_b8", dict(batch=8, use_amp=False)),
    ("eager_amp_cl3d_b8", dict(batch=8, use_amp=True, channels_last=True)),
]:
    try:
        r = bench_raw(model, **kwargs)
        r["label"] = label
        results.append(r)
        print(label, r, flush=True)
    except Exception as e:
        print(label, "FAIL", e, flush=True)
        torch.cuda.empty_cache()

# half weights
print("=== half weights ===", flush=True)
try:
    m_half = type(model)(1, kn=(32, 64, 96, 128, 256), FMU="sub").cuda().eval().half()
    # reload weights
    from hyperdrain.worker import win_to_wsl
    from hyperdrain import config
    ckpt = torch.load(win_to_wsl(config.CKPT), map_location="cuda", weights_only=False)
    # load into fp32 then half
    m_tmp = type(model)(1, kn=(32, 64, 96, 128, 256), FMU="sub").cuda().eval()
    m_tmp.load_state_dict(ckpt["model_weights"], strict=True)
    m_half = m_tmp.half()
    x = torch.randn(8, 1, 20, 64, 64, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        for _ in range(10):
            _ = m_half(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(40):
            a, b = m_half(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    finite = torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
    tps = (40 * 8) / dt
    r = {"label": "weights_fp16_b8", "tps": tps, "ms_tile": 1000 * dt / 320, "finite": finite}
    results.append(r)
    print(r, flush=True)
except Exception as e:
    print("half FAIL", e, flush=True)

# torch.compile modes
print("=== compile ===", flush=True)
for mode in ("default", "reduce-overhead"):
    try:
        # fresh model
        m = type(model)(1, kn=(32, 64, 96, 128, 256), FMU="sub").cuda().eval()
        m.load_state_dict(ckpt["model_weights"], strict=True)
        compiled = torch.compile(m, mode=mode, fullgraph=False)
        r = bench_raw(compiled, batch=8, iters=30, warmup=20, use_amp=True)
        r["label"] = f"compile_{mode}_b8"
        results.append(r)
        print(r["label"], r, flush=True)
    except Exception as e:
        print(f"compile {mode} FAIL", e, flush=True)
        torch.cuda.empty_cache()

# JIT trace
print("=== jit.trace ===", flush=True)
try:
    m = type(model)(1, kn=(32, 64, 96, 128, 256), FMU="sub").cuda().eval()
    m.load_state_dict(ckpt["model_weights"], strict=True)
    example = torch.randn(8, 1, 20, 64, 64, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        traced = torch.jit.trace(m, example, strict=False)
    r = bench_raw(traced, batch=8, iters=40, warmup=15, use_amp=True)
    r["label"] = "jit_trace_b8"
    results.append(r)
    print(r, flush=True)
except Exception as e:
    print("jit FAIL", e, flush=True)

out = ROOT / "experiments/phase6e/HYPERDRAIN/SINGLE_GPU_SPEED_PROBE.json"
out.write_text(json.dumps({"results": results, "params_m": nparams/1e6}, indent=2) + "\n", encoding="utf-8")
print("WROTE", out, flush=True)
best = max(results, key=lambda x: x.get("tps", 0) if x.get("finite", False) else 0)
print("BEST_FINITE", best, flush=True)