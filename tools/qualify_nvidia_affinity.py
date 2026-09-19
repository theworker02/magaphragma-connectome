#!/usr/bin/env python3
"""NVIDIA Affinity production qualification (eager AMP + batch sweep + equivalence).

Sets PRODUCTION_AUTHORIZED=true only when gates pass. Writes receipt under
experiments/phase6e/VAST-AFFINITY-BENCH-001/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

EXPECTED_SHA256 = "73ad8a06229127b6eb9c759e7ba208a3c6a70c2b3556d7742cb6e7d8bd292b92"
CKPT = Path(
    os.environ.get(
        "S7_CHECKPOINT",
        str(REPO / "experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt"),
    )
)
OUT_DIR = Path(os.environ.get("S7_QUAL_OUT", str(REPO / "experiments/phase6e/VAST-AFFINITY-BENCH-001")))
AUTH_FLAG = Path(os.environ.get("S7_PRODUCTION_AUTH_FLAG", "/tmp/affinity_production_authorized"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="16,24,32,48")
    ap.add_argument("--backend", default="eager", choices=["eager", "compile"])
    args = ap.parse_args()

    import numpy as np
    import torch

    from hyperdrain.equivalence import compare_volumes
    from hyperdrain.geometry import CROP_ZYX, STRIDE_ZYX, tile_layout
    from hyperdrain.worker import load_model
    from run_affinity_fullvol_s7_fast_worker import infer_volume_batched

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 2
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    print(f"GPU={name} VRAM={vram_gb:.1f}GiB torch={torch.__version__}", flush=True)
    if vram_gb < 6.0:
        print("FAIL: VRAM < 6 GiB")
        return 2

    if not CKPT.is_file():
        print(f"FAIL: checkpoint missing {CKPT}")
        return 2
    digest = sha256(CKPT)
    if digest != EXPECTED_SHA256:
        print(f"FAIL: checkpoint sha256 mismatch\n got {digest}\n want {EXPECTED_SHA256}")
        return 2
    print("checkpoint sha256 OK", flush=True)

    shape = (64, 256, 256)
    n_tiles = tile_layout(shape, CROP_ZYX, STRIDE_ZYX).n_tiles
    vol = np.random.default_rng(3060).integers(0, 256, size=shape, dtype=np.uint8)

    model, meta, cfg = load_model("eager")
    model.eval()
    torch.backends.cudnn.benchmark = True
    backend_meta = {"applied": "eager", "requested": args.backend}
    if args.backend == "compile":
        from s7_infer_accelerate import accelerate_model

        model, backend_meta = accelerate_model(model, "compile", sample_batch=16)

    ref_aff = ref_bnd = None
    runs = []
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    for batch in batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.perf_counter()
            aff, bnd, tel = infer_volume_batched(model, vol, device="cuda", batch_size=batch)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            finite = bool(np.isfinite(aff).all() and np.isfinite(bnd).all())
            if ref_aff is None:
                ref_aff, ref_bnd = aff, bnd
                equiv = {"pass": True, "reason": "reference"}
            else:
                equiv = compare_volumes(ref_aff, aff, ref_bnd, bnd)
            tps = n_tiles / wall if wall > 0 else 0.0
            row = {
                "batch": batch,
                "tiles_per_second": tps,
                "vram_peak_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "finite": finite,
                "equivalence": equiv,
                "backend": backend_meta.get("applied"),
            }
            runs.append(row)
            print(json.dumps(row), flush=True)
            if not finite or not equiv.get("pass"):
                break
            if len(runs) >= 3:
                tps_l = [r["tiles_per_second"] for r in runs]
                if tps_l[-1] < tps_l[-2] < tps_l[-3]:
                    break
        except RuntimeError as e:
            runs.append({"batch": batch, "error": str(e)})
            print(f"batch {batch} FAIL {e}", flush=True)
            break

    ok = [r for r in runs if r.get("finite") and r.get("equivalence", {}).get("pass") and r.get("tiles_per_second")]
    best = max(ok, key=lambda r: r["tiles_per_second"]) if ok else None
    authorized = bool(best and best["batch"] >= 8)
    receipt = {
        "id": "VAST_NVIDIA_QUALIFICATION",
        "created_at": _now(),
        "gpu": name,
        "vram_gb": vram_gb,
        "torch": torch.__version__,
        "checkpoint_sha256": digest,
        "runs": runs,
        "best": best,
        "PRODUCTION_AUTHORIZED": authorized,
        "crop_zyx": list(CROP_ZYX),
        "stride_zyx": list(STRIDE_ZYX),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "VAST_NVIDIA_QUALIFICATION.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    # merge into equivalence receipts
    eq_path = OUT_DIR / "VAST_EQUIVALENCE_RECEIPTS.json"
    eq = {"id": "VAST_EQUIVALENCE_RECEIPTS", "receipts": []}
    if eq_path.exists():
        eq = json.loads(eq_path.read_text(encoding="utf-8"))
    eq.setdefault("receipts", []).append(receipt)
    eq_path.write_text(json.dumps(eq, indent=2) + "\n", encoding="utf-8")

    if authorized:
        AUTH_FLAG.write_text(
            json.dumps({"PRODUCTION_AUTHORIZED": True, "best": best, "at": _now()}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"PRODUCTION_AUTHORIZED=true best_batch={best['batch']} tps={best['tiles_per_second']:.2f}")
        print(f"WROTE {path} flag={AUTH_FLAG}")
        return 0
    AUTH_FLAG.unlink(missing_ok=True)
    print("PRODUCTION_AUTHORIZED=false")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
