"""Long-lived HyperDrain worker: init once, drain many chunks."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np

from hyperdrain import backends, config, geometry, pipeline, queue, vram
from hyperdrain.packed import infer_volume_packed
from hyperdrain.source_cache import SourceRegionCache
from hyperdrain.mass_production import load_packed_config
from hyperdrain.equivalence import performance_receipt


def win_to_wsl(path: Path | str) -> Path:
    text = str(path).replace("\\", "/")
    on_linux = sys.platform.startswith("linux")
    if text.startswith("/mnt/") and not on_linux:
        parts = text.split("/")
        return Path(f"{parts[2].upper()}:/" + "/".join(parts[3:]))
    if len(text) >= 2 and text[1] == ":" and on_linux:
        return Path(f"/mnt/{text[0].lower()}/{text[3:]}")
    return Path(text)


def fetch_raw(bounds: dict) -> tuple[np.ndarray, str]:
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    z0, z1 = bounds["z"]
    sx, sy, sz = x1 - x0, y1 - y0, z1 - z0
    url = config.DVID_BASE + f"{sx}_{sy}_{sz}/{x0}_{y0}_{z0}"
    expected = sx * sy * sz
    with urlopen(url, timeout=600) as response:
        payload = response.read()
    if len(payload) != expected:
        raise RuntimeError(f"bad payload {len(payload)} != {expected}")
    return np.frombuffer(payload, dtype=np.uint8).reshape((sz, sy, sx)), url


def load_model(backend: str) -> tuple[Any, dict, Any]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm required for HyperDrain worker")
    sys.path.insert(0, str(config.SEGNEURON_TRAIN))
    from model.Mnet import MNet

    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub")
    ckpt = torch.load(win_to_wsl(config.CKPT), map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_weights"], strict=True)
    model = model.cuda().eval()
    model, meta = backends.accelerate(model, backend)
    import os
    if os.environ.get("HYPERDRAIN_SKIP_VRAM_SEARCH") == "1":
        from hyperdrain.vram import VramConfig
        import torch
        cfg = VramConfig(
            batch_size=int(os.environ.get("S7_TILE_BATCH", "16")),
            macro_zyx=(40, 64, 64),
            backend=meta.get("applied", backend),
            precision="fp16",
            tiles_per_sec=0.0,
            vram_peak_bytes=0,
            headroom_frac=0.12,
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            torch_version=torch.__version__,
            hip_version=getattr(torch.version, "hip", None),
            model_hash="skipped",
            stable=False,
        )
        print("hyperdrain: skipped VRAM search", flush=True)
    else:
        cfg = vram.optimize(model, meta.get("applied", backend), win_to_wsl(config.CKPT), force=False)
    return model, meta, cfg


def atomic_write_chunk(
    chunk_dir: Path,
    affinities: np.ndarray,
    boundaries: np.ndarray,
    receipt: dict,
) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    tmp = chunk_dir / ".tmp"
    tmp.mkdir(exist_ok=True)
    aff_t = tmp / "affinities_core_czyx.npy"
    bnd_t = tmp / "boundaries_core.tif"
    rec_t = tmp / "receipt.json"
    np.save(aff_t, affinities, allow_pickle=False)
    import tifffile

    tifffile.imwrite(bnd_t, boundaries)
    rec_t.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # verify
    assert aff_t.stat().st_size > 0
    # rename into place
    (chunk_dir / "affinities_core_czyx.npy").unlink(missing_ok=True)
    (chunk_dir / "boundaries_core.tif").unlink(missing_ok=True)
    aff_t.replace(chunk_dir / "affinities_core_czyx.npy")
    bnd_t.replace(chunk_dir / "boundaries_core.tif")
    rec_t.replace(chunk_dir / "receipt.json")
    try:
        tmp.rmdir()
    except OSError:
        pass


def process_chunk(
    chunk: dict,
    model: Any,
    cfg: Any,
    *,
    mode: str = "tiled",
    prefetched: tuple[np.ndarray, str] | None = None,
    backend_meta: dict | None = None,
) -> dict:
    cid = chunk["id"]
    chunk_dir = config.OUT / "chunks" / cid
    if (chunk_dir / "affinities_core_czyx.npy").exists():
        raise FileExistsError(chunk_dir)
    core = chunk["core_bounds_xyz"]
    phases: dict[str, float | bool] = {}
    wall0 = time.perf_counter()
    if prefetched is not None:
        raw, url = prefetched
        phases["dvid_fetch_seconds"] = 0.0
        phases["dvid_prefetch_hit"] = True
    else:
        t = time.perf_counter()
        raw, url = fetch_raw(core)
        phases["dvid_fetch_seconds"] = time.perf_counter() - t
        phases["dvid_prefetch_hit"] = False

    progress = chunk_dir / "progress.json"
    if mode == "dense":
        aff, bnd, tel = pipeline.infer_volume_macro(model, raw, macro_zyx=tuple(cfg.macro_zyx))
    elif mode in {"packed", "production"}:
        pack_cfg = load_packed_config()
        pack = int(getattr(cfg, "batch_size", None) or pack_cfg.get("pack_size") or 16)
        aff, bnd, tel = infer_volume_packed(
            model,
            raw,
            pack_size=pack,
            prefetch=bool(pack_cfg.get("prefetch", True)),
            progress_path=progress,
            resume_from=0,
        )
    else:
        aff, bnd, tel = pipeline.infer_volume_tiled(
            model,
            raw,
            batch_size=int(cfg.batch_size),
            use_amp=True,
            use_pinned=False,
        )
    phases["affinity_infer_seconds"] = float(tel["infer_seconds"])
    t = time.perf_counter()
    receipt = {
        "chunk_id": cid,
        "engine": "hyperdrain",
        "backend": backend_meta or {},
        "vram_config": {
            "batch_size": cfg.batch_size,
            "macro_zyx": list(cfg.macro_zyx),
            "precision": cfg.precision,
        },
        "source_url": url,
        "core_bounds_xyz": core,
        "throughput_telemetry": {**tel, "phase_seconds": phases},
        "status": queue.AFFINITY_DONE,
        "classification": "MACHINE_PSEUDOLABEL_CHUNK",
    }
    atomic_write_chunk(chunk_dir, aff, bnd, receipt)
    phases["core_crop_save_seconds"] = time.perf_counter() - t
    phases["total_wall_seconds"] = time.perf_counter() - wall0
    receipt["throughput_telemetry"]["phase_seconds"] = phases
    (chunk_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (chunk_dir / "throughput_telemetry.json").write_text(
        json.dumps(receipt["throughput_telemetry"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def run_worker(
    *,
    backend: str = "auto",
    max_chunks: int = 64,
    loop: bool = True,
    mode: str = "tiled",
    experimental_migraphx: bool = False,
) -> int:
    if backend == "migraphx" and not experimental_migraphx:
        print("MIGraphX is experimental — refusing production worker without --experimental-migraphx", flush=True)
        return 2
    if json.loads(config.S6.read_text(encoding="utf-8")).get("status") != "APPROVED":
        raise SystemExit("S6 not approved")

    config.ensure_out()
    chunks = json.loads(config.CHUNKS.read_text(encoding="utf-8"))["chunks"]
    wid = queue.worker_id()
    resolved = backends.normalize_backend(backend)
    if resolved == "auto":
        resolved = backends.resolve_auto()
    print(f"hyperdrain worker_id={wid} backend={resolved} mode={mode}", flush=True)

    eff_mode = mode
    if resolved == "packed" or mode in {"packed", "production"}:
        eff_mode = "packed"
        resolved_load = "eager"
    else:
        resolved_load = resolved
    model, meta, cfg = load_model(resolved_load)
    if eff_mode == "packed":
        pack_cfg = load_packed_config()
        if int(pack_cfg.get("pack_size") or 0) > 0:
            cfg.batch_size = int(pack_cfg["pack_size"])
        meta = {**meta, "applied": "packed-eager", "pack_size": int(cfg.batch_size)}
    print(
        f"backend_meta={json.dumps(meta)} vram_batch={cfg.batch_size} macro={cfg.macro_zyx} mode={eff_mode}",
        flush=True,
    )
    mode = eff_mode
    try:
        import torch as _torch
        queue.register_worker(
            {
                "worker_id": wid,
                "gpu": _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None,
                "backend": meta.get("applied"),
                "pack_size": int(cfg.batch_size),
                "epoch": time.time(),
            }
        )
    except Exception as _exc:  # noqa: BLE001
        print(f"hyperdrain: worker registry skip: {_exc}", flush=True)

    pref = pipeline.PrefetchPipeline(fetch_raw)
    total_done = []
    try:
        while True:
            pending = queue.pending_chunks(chunks)
            print(f"hyperdrain pending={len(pending)} max={max_chunks}", flush=True)
            if not pending:
                break
            batch = pending[:max_chunks]
            for idx, chunk in enumerate(batch):
                cid = chunk["id"]
                if not queue.try_claim(cid, wid=wid):
                    print(f"  skip {cid} (claim lost)", flush=True)
                    continue
                try:
                    use = pref.take()
                    if idx + 1 < len(batch):
                        pref.kick(batch[idx + 1]["core_bounds_xyz"])
                    queue.heartbeat(cid, wid)
                    receipt = process_chunk(
                        chunk, model, cfg, mode=mode, prefetched=use, backend_meta=meta
                    )
                    queue.complete(cid, queue.AFFINITY_DONE)
                    total_done.append(cid)
                    tel = receipt["throughput_telemetry"]
                    print(
                        json.dumps(
                            {
                                "chunk": cid,
                                "tiles_per_sec": tel.get("tiles_per_second"),
                                "infer_s": tel.get("infer_seconds"),
                                "prefetch_hit": tel.get("phase_seconds", {}).get("dvid_prefetch_hit"),
                                "mode": tel.get("mode"),
                            }
                        ),
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"FAILED {cid}: {exc}", flush=True)
                    queue.fail(cid, permanent=False, error=str(exc))
            if not loop:
                break
    finally:
        pref.shutdown()
    print(json.dumps(queue.status_summary(), indent=2), flush=True)
    return 0
