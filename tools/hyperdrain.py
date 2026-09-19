#!/usr/bin/env python3
"""HyperDrain CLI — benchmark | qualify | worker | status | auto."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python tools/hyperdrain.py` and package imports
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from hyperdrain import backends, config, geometry, queue  # noqa: E402
from hyperdrain.equivalence import performance_receipt, run_equivalence  # noqa: E402
from hyperdrain.worker import load_model, win_to_wsl  # noqa: E402


def cmd_benchmark(args: argparse.Namespace) -> int:
    import time

    import numpy as np
    import torch

    from hyperdrain import pipeline

    config.ensure_out()
    audit = geometry.write_audit(config.AUDIT_PATH)
    print(json.dumps({"redundancy_before": audit["redundancy_before"], "n_tiles": audit["n_tiles"]}, indent=2))

    backend = backends.normalize_backend(args.backend)
    if backend == "auto":
        backend = "eager"
    model, meta, cfg = load_model(backend)
    # Synthetic volume matching core shape for local bench (no DVID).
    rng = np.random.default_rng(0)
    volume = rng.integers(0, 256, size=geometry.CORE_SHAPE_ZYX, dtype=np.uint8)

    measurements = []
    for mode in args.modes.split(","):
        mode = mode.strip()
        t0 = time.perf_counter()
        if mode == "dense":
            aff, bnd, tel = pipeline.infer_volume_macro(model, volume, macro_zyx=tuple(cfg.macro_zyx))
        else:
            aff, bnd, tel = pipeline.infer_volume_tiled(model, volume, batch_size=int(cfg.batch_size))
        wall = time.perf_counter() - t0
        meas = {
            "backend": meta.get("applied", backend),
            "mode": mode,
            "tiles_per_second": tel.get("tiles_per_second"),
            "infer_seconds": tel.get("infer_seconds"),
            "wall_seconds": wall,
            "tile_batch_size": tel.get("tile_batch_size"),
            "vram_peak_bytes": tel.get("vram_peak_bytes"),
            "preserved_baseline_eager": config.BASELINE_EAGER_TILES_PER_SEC,
            "preserved_baseline_compile": config.BASELINE_COMPILE_TILES_PER_SEC,
            "shape_aff": list(aff.shape),
        }
        measurements.append(meas)
        print(json.dumps(meas, indent=2), flush=True)

    performance_receipt(measurements=measurements, notes="hyperdrain benchmark synthetic core volume")
    print(f"wrote {config.PERF_RECEIPT}", flush=True)
    print(f"wrote {config.AUDIT_PATH}", flush=True)
    return 0


def cmd_qualify(args: argparse.Namespace) -> int:
    """Run eager reference vs candidate on the same volume; gate production."""
    import numpy as np

    from hyperdrain import pipeline

    config.ensure_out()
    geometry.write_audit(config.AUDIT_PATH)
    rng = np.random.default_rng(42)
    # Smaller volume for qualification speed while preserving tile math
    vol = rng.integers(0, 256, size=(64, 256, 256), dtype=np.uint8)

    ref_model, ref_meta, ref_cfg = load_model("eager")
    ref_aff, ref_bnd, ref_tel = pipeline.infer_volume_tiled(
        ref_model, vol, batch_size=min(8, int(ref_cfg.batch_size))
    )

    cand_backend = backends.normalize_backend(args.backend)
    if cand_backend == "auto":
        cand_backend = "compile"
    if cand_backend == "migraphx" and not args.experimental_migraphx:
        print("Refuse to qualify migraphx without --experimental-migraphx", flush=True)
        return 2

    cand_model, cand_meta, cand_cfg = load_model(cand_backend)
    if args.mode == "dense":
        cand_aff, cand_bnd, cand_tel = pipeline.infer_volume_macro(
            cand_model, vol, macro_zyx=tuple(cand_cfg.macro_zyx)
        )
    else:
        cand_aff, cand_bnd, cand_tel = pipeline.infer_volume_tiled(
            cand_model, vol, batch_size=min(8, int(cand_cfg.batch_size))
        )

    import torch

    versions = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    receipt = run_equivalence(
        ref_name="eager",
        cand_name=f"{cand_meta.get('applied', cand_backend)}:{args.mode}",
        ref_aff=ref_aff,
        cand_aff=cand_aff,
        ref_bnd=ref_bnd,
        cand_bnd=cand_bnd,
        ref_meta=ref_tel,
        cand_meta=cand_tel,
        geometry={"volume": list(vol.shape), "crop": list(geometry.CROP_ZYX), "stride": list(geometry.STRIDE_ZYX)},
        versions=versions,
    )
    print(json.dumps({"pass": receipt["pass"], "comparison": receipt["comparison"]}, indent=2), flush=True)

    q = backends.load_qualified()
    applied = cand_meta.get("applied", cand_backend)
    if receipt["pass"] and applied not in ("migraphx",):
        q.setdefault("qualified", [])
        if applied not in q["qualified"]:
            q["qualified"].append(applied)
        # production = fastest measured among qualified in this run
        tps = float(cand_tel.get("tiles_per_second") or 0)
        ref_tps = float(ref_tel.get("tiles_per_second") or 0)
        if tps >= ref_tps:
            q["production"] = applied
        q["last_qualify"] = {
            "candidate": applied,
            "tiles_per_sec": tps,
            "ref_tiles_per_sec": ref_tps,
            "pass": True,
        }
        backends.save_qualified(q)
        print(f"updated {config.QUALIFIED_MANIFEST}", flush=True)
    else:
        print("NOT promoting to production (FAIL or experimental)", flush=True)
    performance_receipt(
        measurements=[
            {"role": "reference", "backend": "eager", **{k: ref_tel.get(k) for k in ("tiles_per_second", "infer_seconds")}},
            {"role": "candidate", "backend": applied, **{k: cand_tel.get(k) for k in ("tiles_per_second", "infer_seconds")}},
        ],
        notes="qualify run",
    )
    return 0 if receipt["pass"] else 1


def cmd_worker(args: argparse.Namespace) -> int:
    from hyperdrain.worker import run_worker

    return run_worker(
        backend=args.backend,
        max_chunks=args.max_chunks,
        loop=args.loop,
        mode=args.mode,
        experimental_migraphx=args.experimental_migraphx,
    )


def cmd_status(_args: argparse.Namespace) -> int:
    config.ensure_out()
    if not config.QUEUE_STATE.exists():
        queue.bootstrap_from_s7()
    print(queue.format_mass_status())
    summary = queue.status_summary()
    fleet = queue.fleet_metrics()
    print(json.dumps({"summary": summary, "fleet": fleet}, indent=2))
    if config.QUALIFIED_MANIFEST.exists():
        print("qualified:", config.QUALIFIED_MANIFEST.read_text(encoding="utf-8"))
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    """Audit → benchmark eager/compile → qualify compile → optionally start worker."""
    ns = argparse.Namespace(backend="eager", modes="tiled,dense")
    cmd_benchmark(ns)
    q = argparse.Namespace(backend="compile", mode="tiled", experimental_migraphx=False)
    rc = cmd_qualify(q)
    if rc != 0:
        print("auto: qualify failed — not starting production worker", flush=True)
        return rc
    if args.start_worker:
        return cmd_worker(
            argparse.Namespace(
                backend="auto",
                max_chunks=args.max_chunks,
                loop=True,
                mode="tiled",
                experimental_migraphx=False,
            )
        )
    print("auto: qualify PASS; run `hyperdrain.py worker` to drain", flush=True)
    return 0



def cmd_validate_dense(args: argparse.Namespace) -> int:
    """Dense/macro must PASS equivalence AND beat baseline wall_s to be called faster."""
    import numpy as np
    from hyperdrain.speedup import run_dense_speedup_validation, SPEEDUP_VALIDATED_PATH, DENSE_EXPERIMENTAL_PATH
    from hyperdrain.worker import load_model

    config.ensure_out()
    # Modest volume: enough tiles to be meaningful, finishes in reasonable time.
    # Shape divisible-ish for both (20,64,64)@(10,64,64) and macros.
    z, y, x = args.z, args.y, args.x
    rng = np.random.default_rng(0)
    volume = rng.integers(0, 256, size=(z, y, x), dtype=np.uint8)
    macro = tuple(int(part) for part in str(args.macro).split(","))
    print(f"validate-dense volume={(z,y,x)} macro={macro}", flush=True)

    base_model, base_meta, base_cfg = load_model("eager")
    # Same weights/backend for fair dense geometry compare (dense is a pipeline mode).
    cand_model = base_model
    result = run_dense_speedup_validation(
        base_model,
        cand_model,
        volume,
        macro_zyx=macro,
        baseline_batch=int(args.batch),
    )
    print(json.dumps({
        "verdict": result["verdict"],
        "SPEEDUP_VALIDATED": result["SPEEDUP_VALIDATED"],
        "dense_label": result["dense_label"],
        "wall_seconds_baseline": result["wall_seconds_baseline"],
        "wall_seconds_candidate": result["wall_seconds_candidate"],
        "wall_speedup_factor": result["wall_speedup_factor"],
        "equivalence_pass": result["equivalence"].get("pass"),
        "equivalence_reason": result["equivalence"].get("reason") or result["equivalence"].get("stats"),
        "experimental_receipt": str(DENSE_EXPERIMENTAL_PATH),
        "validated_receipt": str(SPEEDUP_VALIDATED_PATH) if result["SPEEDUP_VALIDATED"] else None,
    }, indent=2), flush=True)
    return 0 if result["SPEEDUP_VALIDATED"] else 1



def cmd_mass_benchmark(args: argparse.Namespace) -> int:
    """Warm packed vs eager wall-clock on a representative volume; promotion gate."""
    import numpy as np
    from hyperdrain.mass_production import run_mass_benchmark
    from hyperdrain.worker import load_model

    config.ensure_out()
    z, y, x = args.z, args.y, args.x
    rng = np.random.default_rng(0)
    volume = rng.integers(0, 256, size=(z, y, x), dtype=np.uint8)
    print(f"mass-benchmark volume={(z, y, x)}", flush=True)
    model, meta, cfg = load_model("eager")
    model_c = None
    if args.try_compile:
        try:
            model_c, _, _ = load_model("compile")
        except Exception as exc:  # noqa: BLE001
            print(f"compile unavailable: {exc}", flush=True)
            model_c = None
    receipt = run_mass_benchmark(
        model, volume, also_compile=bool(args.try_compile and model_c is not None), model_compile=model_c
    )
    print(json.dumps({
        "verdict": receipt["verdict"],
        "MASS_PRODUCTION_SPEEDUP_VALIDATED": receipt["speedup"]["MASS_PRODUCTION_SPEEDUP_VALIDATED"],
        "reference_wall": receipt["reference"]["chunk_wall_seconds"],
        "packed_wall": receipt["selected"]["chunk_wall_seconds"],
        "pack_size": receipt["selected"]["pack_size"],
        "equivalence_pass": receipt["equivalence"].get("pass"),
    }, indent=2), flush=True)
    return 0 if receipt["speedup"]["MASS_PRODUCTION_SPEEDUP_VALIDATED"] else 1


def cmd_production(args: argparse.Namespace) -> int:
    """One-command mass production: load packed config, persistent worker, drain queue."""
    from hyperdrain.mass_production import load_packed_config
    from hyperdrain.worker import run_worker

    config.ensure_out()
    if not config.QUEUE_STATE.exists():
        queue.bootstrap_from_s7()
    cfg = load_packed_config()
    print(json.dumps({"production": "start", "packed_config": {
        "pack_size": cfg.get("pack_size"), "prefetch": cfg.get("prefetch"), "backend": cfg.get("backend")
    }}, indent=2), flush=True)
    return run_worker(
        backend="packed",
        max_chunks=int(args.max_chunks),
        loop=not args.once,
        mode="packed",
        experimental_migraphx=False,
    )



def cmd_fleet(args: argparse.Namespace) -> int:
    """Project / launch fleet scaling — the real path to 5×–30× calendar time."""
    from hyperdrain.fleet import format_fleet_report, launch_local_workers, project_fleet

    config.ensure_out()
    plan = project_fleet(int(args.gpus))
    print(format_fleet_report(int(args.gpus)))
    print(json.dumps(plan, indent=2), flush=True)
    if args.launch:
        if int(args.gpus) > 1:
            print(
                "WARNING: launching >1 worker on one GPU causes contention. "
                "For 25–30×, run one `production` worker on each of 25–30 GPU hosts.",
                flush=True,
            )
        launched = launch_local_workers(int(args.launch_workers or args.gpus), max_chunks=int(args.max_chunks))
        print(json.dumps(launched, indent=2), flush=True)
    return 0

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hyperdrain", description="HyperDrain affinity inference engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="Redundancy audit + throughput measurements")
    b.add_argument("--backend", default="eager")
    b.add_argument("--modes", default="tiled,dense", help="Comma list: tiled,dense")
    b.set_defaults(func=cmd_benchmark)

    q = sub.add_parser("qualify", help="Equivalence gate vs frozen eager reference")
    q.add_argument("--backend", default="compile")
    q.add_argument("--mode", default="tiled", choices=["tiled", "dense"])
    q.add_argument("--experimental-migraphx", action="store_true")
    q.set_defaults(func=cmd_qualify)

    w = sub.add_parser("worker", help="Long-lived queue worker")
    w.add_argument("--backend", default="auto")
    w.add_argument("--mode", default="tiled", choices=["tiled", "dense", "packed", "production"])
    w.add_argument("--max-chunks", type=int, default=64)
    w.add_argument("--loop", action="store_true", default=True)
    w.add_argument("--no-loop", action="store_false", dest="loop")
    w.add_argument("--experimental-migraphx", action="store_true")
    w.set_defaults(func=cmd_worker)

    s = sub.add_parser("status", help="Queue + qualification status")
    s.set_defaults(func=cmd_status)

    a = sub.add_parser("auto", help="Audit, benchmark, qualify; optional worker")
    a.add_argument("--start-worker", action="store_true")
    a.add_argument("--max-chunks", type=int, default=4)
    a.set_defaults(func=cmd_auto)

    vd = sub.add_parser(
        "validate-dense",
        help="Dense/macro vs eager tiled: equivalence + wall_s/chunk (SPEEDUP_VALIDATED gate)",
    )
    vd.add_argument("--macro", default="40,64,64", help="macro Z,Y,X")
    vd.add_argument("--batch", type=int, default=16, help="baseline tiled batch")
    vd.add_argument("--z", type=int, default=64)
    vd.add_argument("--y", type=int, default=256)
    vd.add_argument("--x", type=int, default=256)
    vd.set_defaults(func=cmd_validate_dense)

    mb = sub.add_parser("mass-benchmark", help="Warm packed vs eager; MASS_PRODUCTION_SPEEDUP_VALIDATED gate")
    mb.add_argument("--z", type=int, default=64)
    mb.add_argument("--y", type=int, default=256)
    mb.add_argument("--x", type=int, default=256)
    mb.add_argument("--try-compile", action="store_true")
    mb.set_defaults(func=cmd_mass_benchmark)

    pr = sub.add_parser("production", help="Persistent packed worker draining the mass-production queue")
    pr.add_argument("--max-chunks", type=int, default=64)
    pr.add_argument("--once", action="store_true", help="Process at most max-chunks then exit")
    pr.set_defaults(func=cmd_production)

    fl = sub.add_parser("fleet", help="Project N-GPU calendar speedup (path to 5x-30x wall time)")
    fl.add_argument("--gpus", type=int, default=30, help="Fleet size to project")
    fl.add_argument("--launch", action="store_true", help="Also spawn local workers")
    fl.add_argument("--launch-workers", type=int, default=0)
    fl.add_argument("--max-chunks", type=int, default=64)
    fl.set_defaults(func=cmd_fleet)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
