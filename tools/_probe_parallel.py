import os, sys, time, json, copy
from pathlib import Path

ROOT = Path("/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome")
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "third_party/segneuron/Train_and_Inference"))
os.environ["HYPERDRAIN_SKIP_VRAM_SEARCH"] = "1"
os.chdir(ROOT)

import torch
import torch.nn.functional as F
import numpy as np

from model.Mnet import MNet, FMU, Down, Up
from hyperdrain.worker import load_model, win_to_wsl
from hyperdrain import config, pipeline
from hyperdrain.equivalence import compare_volumes
from hyperdrain.packed import infer_volume_packed

# --- Patch Down/Up to run both branches on parallel streams ---
_orig_down = Down.forward
_orig_up = Up.forward

def down_forward_parallel(self, x):
    if self.downsample:
        if self.mode_in == "both":
            x2d, x3d = x
            p2d = F.max_pool3d(x2d, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            if x3d.shape[2] >= self.min_z:
                p3d = F.max_pool3d(x3d, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            else:
                p3d = F.max_pool3d(x3d, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            x = FMU(p2d, p3d, mode=self.FMU)
        elif self.mode_in == "2d":
            x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        elif self.mode_in == "3d":
            if x.shape[2] >= self.min_z:
                x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            else:
                x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    if self.mode_out == "2d":
        return self.CB2d(x)
    if self.mode_out == "3d":
        return self.CB3d(x)
    if self.mode_out == "both":
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        out2d = out3d = None
        with torch.cuda.stream(s1):
            out2d = self.CB2d(x)
        with torch.cuda.stream(s2):
            out3d = self.CB3d(x)
        torch.cuda.current_stream().wait_stream(s1)
        torch.cuda.current_stream().wait_stream(s2)
        return out2d, out3d
    raise RuntimeError(self.mode_out)

def up_forward_parallel(self, x):
    x2d, xskip2d, x3d, xskip3d = x
    tarSize = xskip2d.shape[2:]
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    with torch.cuda.stream(s1):
        up2d = F.interpolate(x2d, size=tarSize, mode="trilinear", align_corners=False)
    with torch.cuda.stream(s2):
        up3d = F.interpolate(x3d, size=tarSize, mode="trilinear", align_corners=False)
    torch.cuda.current_stream().wait_stream(s1)
    torch.cuda.current_stream().wait_stream(s2)
    cat = torch.cat([FMU(xskip2d, xskip3d, self.FMU), FMU(up2d, up3d, self.FMU)], dim=1)
    if self.mode_out == "2d":
        return self.CB2d(cat)
    if self.mode_out == "3d":
        return self.CB3d(cat)
    if self.mode_out == "both":
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        with torch.cuda.stream(s1):
            o2 = self.CB2d(cat)
        with torch.cuda.stream(s2):
            o3 = self.CB3d(cat)
        torch.cuda.current_stream().wait_stream(s1)
        torch.cuda.current_stream().wait_stream(s2)
        return o2, o3
    raise RuntimeError(self.mode_out)

def bench(m, batch=8, iters=30, warmup=12):
    x = torch.randn(batch, 1, 20, 64, 64, device="cuda")
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast("cuda", dtype=torch.float16):
                _ = m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.autocast("cuda", dtype=torch.float16):
                a, b = m(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    finite = torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
    return (iters * batch) / dt, finite

print("loading...", flush=True)
base, meta, cfg = load_model("eager")
torch.backends.cudnn.benchmark = True

tps0, fin0 = bench(base)
print(f"baseline_eager tps={tps0:.2f} finite={fin0}", flush=True)

# Apply parallel patches
Down.forward = down_forward_parallel
Up.forward = up_forward_parallel
tps1, fin1 = bench(base)
print(f"stream_parallel tps={tps1:.2f} finite={fin1} speedup={tps1/tps0:.3f}", flush=True)

# FMU fp32 + compile attempt
def FMU_fp32(x1, x2, mode="sub"):
    if mode == "sum":
        return torch.add(x1, x2)
    if mode == "sub":
        return torch.abs(x1.float() - x2.float())
    if mode == "cat":
        return torch.cat((x1, x2), dim=1)
    raise Exception(mode)

import model.Mnet as mnet_mod
mnet_mod.FMU = FMU_fp32
# also need FMU in Down/Up closures - they import FMU at call time from module if using global
# Our patched forwards call FMU from this scope - restore serial forwards that use FP32 FMU

Down.forward = _orig_down
Up.forward = _orig_up

# Monkeypatch module-level FMU used by original forward
import model.Mnet as MM
MM.FMU = FMU_fp32

tps2, fin2 = bench(base)
print(f"fmu_fp32_eager tps={tps2:.2f} finite={fin2}", flush=True)

print("compiling with FMU_fp32...", flush=True)
compiled = torch.compile(base, mode="reduce-overhead", fullgraph=False)
tps3, fin3 = bench(compiled, iters=25, warmup=20)
print(f"compile_fmu_fp32 tps={tps3:.2f} finite={fin3} vs_base={tps3/tps0:.3f}", flush=True)

# Equivalence: packed volume small
rng = np.random.default_rng(0)
vol = rng.integers(0, 256, size=(48, 192, 192), dtype=np.uint8)
# restore eager baseline model without compile for ref
Down.forward = _orig_down
Up.forward = _orig_up
MM.FMU = FMU  # original
ref_model, _, _ = load_model("eager")
# candidate = compiled if finite else stream parallel
if fin3:
    cand = compiled
    cand_name = "compile_fmu_fp32"
elif fin1 and tps1 > tps0 * 1.05:
    Down.forward = down_forward_parallel
    Up.forward = up_forward_parallel
    cand = base
    cand_name = "stream_parallel"
else:
    cand = None
    cand_name = "none"

out = {
    "baseline_tps": tps0,
    "stream_parallel": {"tps": tps1, "finite": fin1, "speedup": tps1 / tps0},
    "fmu_fp32_eager": {"tps": tps2, "finite": fin2},
    "compile_fmu_fp32": {"tps": tps3, "finite": fin3, "speedup": tps3 / tps0},
    "selected": cand_name,
}
if cand is not None:
    print("equivalence check...", flush=True)
    torch.cuda.synchronize()
    # warm
    pipeline.infer_volume_tiled(ref_model, vol, batch_size=8)
    pipeline.infer_volume_tiled(cand, vol, batch_size=8)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ra, rb, rt = pipeline.infer_volume_tiled(ref_model, vol, batch_size=8)
    torch.cuda.synchronize()
    ref_wall = time.perf_counter() - t0
    t1 = time.perf_counter()
    ca, cb, ct = pipeline.infer_volume_tiled(cand, vol, batch_size=8)
    torch.cuda.synchronize()
    cand_wall = time.perf_counter() - t1
    cmp = compare_volumes(ra, ca, rb, cb)
    out["equivalence"] = cmp
    out["ref_wall"] = ref_wall
    out["cand_wall"] = cand_wall
    out["wall_speedup"] = ref_wall / cand_wall if cand_wall > 0 else None
    print("equiv", cmp.get("pass"), "wall", ref_wall, "->", cand_wall, "x", out["wall_speedup"], flush=True)

path = ROOT / "experiments/phase6e/HYPERDRAIN/SINGLE_GPU_PARALLEL_PROBE.json"
path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print("WROTE", path, flush=True)
print(json.dumps(out, indent=2), flush=True)