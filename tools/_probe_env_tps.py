import os, sys, time, json
from pathlib import Path

ROOT = Path("/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome")
sys.path.insert(0, str(ROOT / "tools"))
os.environ["HYPERDRAIN_SKIP_VRAM_SEARCH"] = "1"
os.chdir(ROOT)

import torch
from hyperdrain.worker import load_model

label = os.environ.get("PROBE_LABEL", "x")
m, _, _ = load_model("eager")
torch.backends.cudnn.benchmark = True
x = torch.randn(8, 1, 20, 64, 64, device="cuda")
with torch.inference_mode():
    for _ in range(15):
        with torch.autocast("cuda", torch.float16):
            _ = m(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(40):
        with torch.autocast("cuda", torch.float16):
            a, b = m(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
print(
    json.dumps(
        {
            "label": label,
            "tps": 320 / dt,
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        }
    )
)
