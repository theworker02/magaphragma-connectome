#!/usr/bin/env python3
"""Affinity cheapest production-equivalent compute probe.

Benchmarks:
  1) batch-size sweep until aggregate tiles/s stops improving
  2) 1–4 concurrent Affinity workers on one GPU (where VRAM permits)

Preserves FAST production tile math (crop 20x64x64, stride 10x64x64, AMP fp16,
gaussian blend). Equivalence via hyperdrain.compare_volumes gates.

Outputs JSON under experiments/phase6e/HYPERDRAIN/.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import resource
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/mnt/c/Users/matth/OneDrive/Desktop/magaphragma-connectome")
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)
os.environ.setdefault("HYPERDRAIN_SKIP_VRAM_SEARCH", "1")

import numpy as np
import torch

from hyperdrain.equivalence import array_digest, compare_volumes
from hyperdrain.geometry import CROP_ZYX, STRIDE_ZYX, tile_layout
from hyperdrain.worker import load_model
from run_affinity_fullvol_s7_fast_worker import infer_volume_batched

EQ_DIR = Path("/tmp/affinity_cost_density_eq")

OUT = ROOT / "experiments/phase6e/HYPERDRAIN/AFFINITY_COST_DENSITY_PROBE.json"
PROGRESS = ROOT / "experiments/phase6e/AFFINITY-FULLVOL-S7-001/PROGRESS.json"
TILES_PER_CHUNK = 3468  # full core 128x1024x1024 @ FAST stride
# Probe volume: same Z/Y/X tile math, fewer tiles for wall-clock feasibility.
PROBE_SHAPE = (64, 256, 256)  # still production crop/stride
BATCH_SWEEP = [8, 16, 24, 32, 48, 64, 96]
WORKER_COUNTS = [1, 2, 3, 4]
BUDGETS_USD = [25, 50, 100, 160]
SAMPLE_HZ = 2.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _probe_tile_count(shape=PROBE_SHAPE) -> int:
    layout = tile_layout(shape, CROP_ZYX, STRIDE_ZYX)
    counts = layout.counts_zyx
    return int(counts[0] * counts[1] * counts[2])


def _host_ram_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _rss_bytes() -> int:
    # Linux: ru_maxrss is KB
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _sample_rocm() -> dict[str, Any]:
    """Best-effort GPU util / VRAM from rocm-smi JSON."""
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        data = json.loads(out)
        # schema varies; flatten first card
        card = next(iter(data.values())) if isinstance(data, dict) else {}
        if isinstance(card, dict) and len(card) == 1 and isinstance(next(iter(card.values())), dict):
            card = next(iter(card.values()))
        util = None
        for k, v in (card or {}).items():
            lk = str(k).lower()
            if "gpu use" in lk or "gpu%" in lk or lk.endswith("gpu"):
                try:
                    util = float(str(v).replace("%", "").strip())
                except ValueError:
                    pass
        return {"raw": card, "gpu_util_pct": util}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "gpu_util_pct": None}


def _sample_cpu_pct(prev: tuple[float, float] | None) -> tuple[float | None, tuple[float, float]]:
    """Return (cpu_pct since prev, new_state) from /proc/stat."""
    with open("/proc/stat") as f:
        parts = f.readline().split()
    idle = float(parts[4]) + float(parts[5])
    total = sum(float(x) for x in parts[1:8])
    state = (idle, total)
    if prev is None:
        return None, state
    di = idle - prev[0]
    dt = total - prev[1]
    if dt <= 0:
        return None, state
    return 100.0 * (1.0 - di / dt), state


class HostSampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []

    def start(self) -> None:
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        utils = [s["gpu_util_pct"] for s in self.samples if s.get("gpu_util_pct") is not None]
        cpus = [s["cpu_pct"] for s in self.samples if s.get("cpu_pct") is not None]
        return {
            "n_samples": len(self.samples),
            "gpu_util_pct_mean": float(np.mean(utils)) if utils else None,
            "gpu_util_pct_max": float(np.max(utils)) if utils else None,
            "cpu_pct_mean": float(np.mean(cpus)) if cpus else None,
            "cpu_pct_max": float(np.max(cpus)) if cpus else None,
            "host_ram_total_bytes": _host_ram_bytes(),
            "process_rss_peak_bytes": _rss_bytes(),
        }

    def _run(self) -> None:
        cpu_state = None
        while not self._stop.is_set():
            cpu_pct, cpu_state = _sample_cpu_pct(cpu_state)
            roc = _sample_rocm()
            self.samples.append(
                {
                    "t": time.time(),
                    "cpu_pct": cpu_pct,
                    "gpu_util_pct": roc.get("gpu_util_pct"),
                    "torch_mem_allocated": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None,
                    "torch_mem_reserved": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else None,
                }
            )
            self._stop.wait(1.0 / SAMPLE_HZ)


def _make_volume(seed: int, shape=PROBE_SHAPE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _run_one_infer(model, volume, batch: int) -> tuple[np.ndarray, np.ndarray, dict]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    aff, bnd, tel = infer_volume_batched(model, volume, device="cuda", batch_size=batch)
    torch.cuda.synchronize()
    tel["vram_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    return aff, bnd, tel


def _worker_entry(q: mp.Queue, worker_id: int, batch: int, seed: int, n_tiles_expected: int) -> None:
    """Child process: load model, infer one probe volume, report metrics."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    try:
        model, _meta, _cfg = load_model("eager")
        model.eval()
        torch.backends.cudnn.benchmark = True
        vol = _make_volume(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        aff, bnd, tel = infer_volume_batched(model, vol, device="cuda", batch_size=batch)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        EQ_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EQ_DIR / f"w{worker_id}_aff.npy", aff)
        np.save(EQ_DIR / f"w{worker_id}_bnd.npy", bnd)
        q.put(
            {
                "worker_id": worker_id,
                "ok": True,
                "seed": seed,
                "wall_seconds": wall,
                "tiles_total": int(tel["tiles_total"]),
                "tiles_per_second": float(tel["tiles_total"]) / wall if wall > 0 else None,
                "vram_peak_bytes": int(torch.cuda.max_memory_allocated()),
                "infer_seconds": float(tel["infer_seconds"]),
                "aff_digest": array_digest(aff),
                "bnd_digest": array_digest(bnd),
                "n_tiles_expected": n_tiles_expected,
                "finite": bool(np.isfinite(aff).all() and np.isfinite(bnd).all()),
                "aff_shape": list(aff.shape),
                "bnd_shape": list(bnd.shape),
            }
        )
        del aff, bnd
    except Exception as e:
        q.put({"worker_id": worker_id, "ok": False, "error": f"{type(e).__name__}: {e}"})


def bench_batch_sweep(model, ref_vol: np.ndarray) -> dict:
    print("=== batch sweep (single worker, production tile math) ===", flush=True)
    sampler = HostSampler()
    results = []
    ref_aff = ref_bnd = None
    # Warmup
    _ = _run_one_infer(model, ref_vol, 8)

    for batch in BATCH_SWEEP:
        sampler.start()
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            aff, bnd, tel = _run_one_infer(model, ref_vol, batch)
            host = sampler.stop()
            if ref_aff is None:
                ref_aff, ref_bnd = aff, bnd
                equiv = {"pass": True, "reason": "reference_batch", "batch": batch}
            else:
                equiv = compare_volumes(ref_aff, aff, ref_bnd, bnd)
            row = {
                "batch_size": batch,
                "tiles_total": tel["tiles_total"],
                "wall_seconds": tel["infer_seconds"],
                "tiles_per_second": tel["tiles_per_second"],
                "vram_peak_bytes": tel["vram_peak_bytes"],
                "vram_peak_gib": tel["vram_peak_bytes"] / (1024**3) if tel["vram_peak_bytes"] else None,
                "equivalence": equiv,
                "host": host,
                "finite": bool(np.isfinite(aff).all() and np.isfinite(bnd).all()),
            }
            results.append(row)
            print(
                f"  batch={batch} tps={row['tiles_per_second']:.2f} "
                f"vram={row['vram_peak_gib']:.2f}GiB equiv={equiv.get('pass')} "
                f"gpu_util~{host.get('gpu_util_pct_mean')}",
                flush=True,
            )
            # Stop if OOM-like failure already handled; stop improving if last 2 worse
            if len(results) >= 3:
                tps = [r["tiles_per_second"] for r in results if r["tiles_per_second"]]
                if len(tps) >= 3 and tps[-1] < tps[-2] < tps[-3]:
                    print("  plateau detected (two consecutive regressions); stopping sweep", flush=True)
                    break
        except RuntimeError as e:
            sampler.stop()
            results.append({"batch_size": batch, "ok": False, "error": str(e)})
            print(f"  batch={batch} FAIL {e}", flush=True)
            torch.cuda.empty_cache()
            break

    # Best batch among equiv-pass rows
    ok = [r for r in results if r.get("tiles_per_second") and r.get("equivalence", {}).get("pass") and r.get("finite", True)]
    best = max(ok, key=lambda r: r["tiles_per_second"]) if ok else None
    return {"runs": results, "best": best, "reference_batch": BATCH_SWEEP[0]}


def bench_concurrent(best_batch: int, n_tiles: int) -> list[dict]:
    print(f"=== concurrent workers on 1 GPU (batch={best_batch}) ===", flush=True)
    # Spawn context: fork can be unsafe with ROCm; use spawn.
    ctx = mp.get_context("spawn")
    rows = []
    for n in WORKER_COUNTS:
        # Rough VRAM gate: ~3.3 GiB * n must fit with headroom in free VRAM
        free, total = torch.cuda.mem_get_info()
        est = 3.3 * (1024**3) * n
        print(f"  n_workers={n} free_vram={free/(1024**3):.1f}GiB est_need~{est/(1024**3):.1f}GiB", flush=True)
        if est > free * 0.95:
            rows.append(
                {
                    "n_workers": n,
                    "skipped": True,
                    "reason": f"estimated VRAM {est/(1024**3):.1f}GiB exceeds free {free/(1024**3):.1f}GiB",
                }
            )
            print("    SKIP (VRAM)", flush=True)
            continue

        q: mp.Queue = ctx.Queue()
        procs = []
        sampler = HostSampler()
        torch.cuda.empty_cache()
        # Drop parent model tensors before spawning to free VRAM for children
        sampler.start()
        wall0 = time.perf_counter()
        for i in range(n):
            p = ctx.Process(target=_worker_entry, args=(q, i, best_batch, 1000 + i, n_tiles))
            p.start()
            procs.append(p)
        worker_results = []
        for _ in range(n):
            worker_results.append(q.get(timeout=1800))
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
        wall = time.perf_counter() - wall0
        host = sampler.stop()

        ok = all(r.get("ok") for r in worker_results)
        if not ok:
            rows.append({"n_workers": n, "ok": False, "workers": worker_results, "host": host})
            print(f"    FAIL workers={worker_results}", flush=True)
            continue

        # Equivalence: each concurrent worker vs solo eager on the same seed.
        equiv_rows = []
        solo_model, _, _ = load_model("eager")
        solo_model.eval()
        torch.backends.cudnn.benchmark = True
        for wr in sorted(worker_results, key=lambda r: r["worker_id"]):
            seed = int(wr["seed"])
            ref_aff, ref_bnd, _ = _run_one_infer(solo_model, _make_volume(seed), best_batch)
            cand_aff = np.load(EQ_DIR / f"w{wr['worker_id']}_aff.npy")
            cand_bnd = np.load(EQ_DIR / f"w{wr['worker_id']}_bnd.npy")
            eq = compare_volumes(ref_aff, cand_aff, ref_bnd, cand_bnd)
            eq["worker_id"] = wr["worker_id"]
            eq["seed"] = seed
            eq["bitwise_aff"] = array_digest(ref_aff) == wr.get("aff_digest")
            equiv_rows.append(eq)
            print(
                f"    equiv w{wr['worker_id']} pass={eq.get('pass')} "
                f"bitwise={eq['bitwise_aff']} disagree={eq.get('stats', {}).get('decision_disagree_frac')}",
                flush=True,
            )
        del solo_model
        torch.cuda.empty_cache()

        per_tps = [r["tiles_per_second"] for r in worker_results]
        total_tiles = sum(r["tiles_total"] for r in worker_results)
        agg_tps = total_tiles / wall if wall > 0 else None
        vrams = [r["vram_peak_bytes"] for r in worker_results]
        row = {
            "n_workers": n,
            "ok": True,
            "wall_seconds": wall,
            "aggregate_tiles_per_second": agg_tps,
            "per_worker_tiles_per_second": per_tps,
            "sum_per_worker_tps": float(sum(per_tps)),
            "vram_peak_bytes_per_worker": vrams,
            "vram_peak_gib_per_worker": [v / (1024**3) for v in vrams],
            "vram_peak_bytes_sum_reported": int(sum(vrams)),
            "torch_mem_reserved_max_bytes": max(
                (s["torch_mem_reserved"] or 0 for s in sampler.samples), default=None
            ),
            "host": host,
            "workers": worker_results,
            "equivalence": equiv_rows,
            "equivalence_all_pass": all(e.get("pass") for e in equiv_rows),
            "speedup_vs_1_worker": None,
        }
        rows.append(row)
        print(
            f"    wall={wall:.1f}s agg_tps={agg_tps:.2f} "
            f"per_tps={[round(x,1) for x in per_tps]} "
            f"vram_sum_reported={sum(vrams)/(1024**3):.1f}GiB",
            flush=True,
        )

    # Relative speedup vs n=1
    base = next((r for r in rows if r.get("n_workers") == 1 and r.get("aggregate_tiles_per_second")), None)
    if base:
        b = base["aggregate_tiles_per_second"]
        for r in rows:
            if r.get("aggregate_tiles_per_second"):
                r["speedup_vs_1_worker"] = r["aggregate_tiles_per_second"] / b
    return rows


def remaining_workload() -> dict:
    n_total = 28798
    n_done = 0
    if PROGRESS.exists():
        prog = json.loads(PROGRESS.read_text())
        n_total = int(prog.get("n_total", n_total))
        n_done = int(prog.get("n_done", 0))
        n_not = int(prog.get("n_not_started", n_total - n_done))
    else:
        n_not = n_total - n_done
    # Conservatively treat not_started as remaining affinity work
    remaining_chunks = n_not
    remaining_tiles = remaining_chunks * TILES_PER_CHUNK
    return {
        "n_total_chunks": n_total,
        "n_done_chunks": n_done,
        "n_remaining_chunks": remaining_chunks,
        "tiles_per_chunk": TILES_PER_CHUNK,
        "n_remaining_tiles": remaining_tiles,
        "progress_updated_at": json.loads(PROGRESS.read_text()).get("updated_at") if PROGRESS.exists() else None,
    }


def gpu_hours_and_breakeven(agg_tps: float, remaining_tiles: int) -> dict:
    gpu_hours = remaining_tiles / agg_tps / 3600.0 if agg_tps > 0 else None
    be = {}
    for b in BUDGETS_USD:
        be[str(b)] = (b / gpu_hours) if gpu_hours and gpu_hours > 0 else None
    return {
        "required_gpu_hours": gpu_hours,
        "break_even_usd_per_gpu_hour": be,
        "note": "break_even = budget / required_gpu_hours (max payable $/GPU-h to finish within budget)",
    }


def main() -> None:
    print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None), flush=True)
    assert torch.cuda.is_available(), "CUDA/ROCm device required"
    name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"device={name} total_vram={total_mem/(1024**3):.1f}GiB", flush=True)

    n_tiles = _probe_tile_count()
    print(f"probe_shape={PROBE_SHAPE} tiles={n_tiles} crop={CROP_ZYX} stride={STRIDE_ZYX}", flush=True)

    model, meta, cfg = load_model("eager")
    model.eval()
    torch.backends.cudnn.benchmark = True
    ref_vol = _make_volume(42)

    batch_report = bench_batch_sweep(model, ref_vol)
    best_batch = int(batch_report["best"]["batch_size"]) if batch_report["best"] else 16
    best_tps = float(batch_report["best"]["tiles_per_second"]) if batch_report["best"] else None

    # Free parent model before multi-process concurrent test
    del model
    torch.cuda.empty_cache()
    concurrent = bench_concurrent(best_batch, n_tiles)

    workload = remaining_workload()
    # Choose config: max aggregate TPS among successful concurrent runs; else best single batch
    candidates = [r for r in concurrent if r.get("ok") and r.get("aggregate_tiles_per_second")]
    if candidates:
        chosen = max(candidates, key=lambda r: r["aggregate_tiles_per_second"])
        chosen_tps = chosen["aggregate_tiles_per_second"]
        chosen_desc = f"{chosen['n_workers']}x concurrent workers @ batch {best_batch} on 1 GPU"
    else:
        chosen = {"n_workers": 1, "aggregate_tiles_per_second": best_tps, "batch_size": best_batch}
        chosen_tps = best_tps
        chosen_desc = f"1 worker @ batch {best_batch}"

    economics = gpu_hours_and_breakeven(chosen_tps or 0.0, workload["n_remaining_tiles"])

    # Also report economics at single-worker best for comparison
    econ_1 = gpu_hours_and_breakeven(best_tps or 0.0, workload["n_remaining_tiles"]) if best_tps else None

    report = {
        "id": "AFFINITY_COST_DENSITY_PROBE",
        "created_at": _now(),
        "device": {"name": name, "total_vram_bytes": total_mem, "torch": torch.__version__},
        "tile_math": {
            "crop_zyx": list(CROP_ZYX),
            "stride_zyx": list(STRIDE_ZYX),
            "amp": "float16",
            "infer_backend": "eager",
            "probe_shape_zyx": list(PROBE_SHAPE),
            "probe_tiles": n_tiles,
            "production_tiles_per_chunk": TILES_PER_CHUNK,
        },
        "equivalence_gate": {
            "fn": "hyperdrain.equivalence.compare_volumes",
            "abs_tol": 1e-3,
            "decision_threshold": 0.5,
            "max_decision_disagree_frac": 1e-4,
            "bitwise": False,
        },
        "batch_sweep": batch_report,
        "concurrent_workers": concurrent,
        "remaining_workload": workload,
        "recommended": {
            "description": chosen_desc,
            "batch_size": best_batch,
            "n_workers_on_gpu": chosen.get("n_workers"),
            "aggregate_tiles_per_second": chosen_tps,
            "economics": economics,
            "economics_single_worker_best_batch": econ_1,
        },
        "cheapest_config_guidance": {
            "vram_floor_gib": 3.3,
            "prefer": "cheapest CUDA/ROCm GPU with >= ~4–6 GiB usable VRAM for 1 worker; "
            "or enough VRAM for N concurrent workers only if aggregate TPS rises",
            "do_not_prefer": "extra VRAM, prestige SKUs, multi-GPU hosts unless $/tile falls",
            "aws_examples_descending_cost_bias": [
                "spot g4dn.xlarge (T4 16GB) if available and equiv holds",
                "spot g5.xlarge / g6.xlarge only if cheaper Spot $/tile than T4 class",
                "avoid g6.2xlarge — wasted CPU quota vs 1x GPU",
            ],
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"WROTE {OUT}", flush=True)
    print(
        f"RECOMMENDED {chosen_desc} agg_tps={chosen_tps:.2f} "
        f"gpu_hours={economics['required_gpu_hours']:.1f} "
        f"breakeven[$25]={economics['break_even_usd_per_gpu_hour']['25']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    # Avoid accidental fork bomb under nested runners
    mp.freeze_support()
    main()
