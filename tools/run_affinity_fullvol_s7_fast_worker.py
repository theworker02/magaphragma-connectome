"""FAST S7 worker under AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_001.

Batched tiles, core-only fetch, stride (10,64,64), AMP, no raw persist, seg deferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from urllib.request import urlopen

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
CONTRACT_FAST_001 = REPO / "experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_001.json"
CONTRACT_FAST_002 = REPO / "experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_002.json"
S6 = REPO / "experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json"
CHUNKS = Path(
    os.environ.get(
        "S7_CHUNKS",
        str(REPO / "local_research_build/phase5c-production/chunks.json"),
    )
)
ROI_PKG = REPO / "experiments/phase6e/AFFINITY-ROI-001/package/ROI_PACKAGE.json"
OUT = Path(os.environ.get("S7_OUT", str(REPO / "experiments/phase6e/AFFINITY-FULLVOL-S7-001")))
CKPT = Path(
    os.environ.get(
        "S7_CHECKPOINT",
        str(REPO / "experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt"),
    )
)
BASE = "https://waspem-dvid2.flatironinstitute.org/api/node/aa49d16e88424cdbae89f4b8ced2a49b/five_yuri_4contrast/raw/0_1_2/"
CROP_SIZE = (20, 64, 64)
STRIDE = (10, 64, 64)
# FAST_002 default 16. Batch 32 regressed tiles/sec on ROCm 7.2 WSL; override via S7_TILE_BATCH.
BATCH = int(os.environ.get("S7_TILE_BATCH", "16"))
STATE_PATH = OUT / "queue_state.json"
N_CHUNKS = 28798
# Stage large writes on Linux tmpfs/ext4 then move onto the Windows mount (OneDrive /mnt/c is slow).
STAGE_DIR = Path(os.environ.get("S7_STAGE_DIR", "/tmp/s7-fast-stage"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def win_to_wsl(path: Path | str) -> Path:
    text = str(path).replace("\\", "/")
    on_linux = sys.platform.startswith("linux")
    if text.startswith("/mnt/") and not on_linux:
        parts = text.split("/")
        return Path(f"{parts[2].upper()}:/" + "/".join(parts[3:]))
    if len(text) >= 2 and text[1] == ":" and on_linux:
        return Path(f"/mnt/{text[0].lower()}/{text[3:]}")
    return Path(text)


def overlaps(a: dict, b: dict) -> bool:
    for ax in ("x", "y", "z"):
        if a[ax][1] <= b[ax][0] or b[ax][1] <= a[ax][0]:
            return False
    return True


def roi_bounds_xyz(pkg: dict) -> dict:
    ox, oy, oz = pkg["roi_origin_xyz"]
    sx, sy, sz = pkg["roi_size_xyz"]
    return {"x": [ox, ox + sx], "y": [oy, oy + sy], "z": [oz, oz + sz]}


def load_state(chunks: list[dict]) -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "created_at": _now(),
        "contract_id": "AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_001",
        "completed": [],
        "failed": [],
        "status_by_id": {c["id"]: "NOT_STARTED" for c in chunks},
    }


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(STATE_PATH)


def chunk_shard_index(chunk_id: str) -> int:
    """Stable shard from trailing digits of MV-CHUNK-########."""
    digits = "".join(ch for ch in chunk_id if ch.isdigit())
    return int(digits[-6:] if digits else "0")


def priority_queue(chunks: list[dict], state: dict, roi: dict, shard: tuple[int, int] | None = None) -> list[dict]:
    by_id = {c["id"]: c for c in chunks}
    # Reclaim RUNNING stubs that never wrote affinities (crashed/killed workers)
    for c in chunks:
        cid = c["id"]
        if state["status_by_id"].get(cid) == "RUNNING":
            aff = OUT / "chunks" / cid / "affinities_core_czyx.npy"
            if not aff.exists():
                state["status_by_id"][cid] = "NOT_STARTED"
    pending = [c for c in chunks if state["status_by_id"].get(c["id"]) == "NOT_STARTED"]
    if shard is not None:
        si, sn = shard
        pending = [c for c in pending if chunk_shard_index(c["id"]) % sn == si]
    seed = [c for c in pending if overlaps(c["core_bounds_xyz"], roi)]
    ordered, seen, frontier = [], set(), [c["id"] for c in sorted(seed, key=lambda c: c["id"])]
    while frontier:
        cid = frontier.pop(0)
        if cid in seen or state["status_by_id"].get(cid) != "NOT_STARTED":
            continue
        if shard is not None:
            si, sn = shard
            if chunk_shard_index(cid) % sn != si:
                continue
        seen.add(cid)
        ordered.append(by_id[cid])
        for nid in by_id[cid].get("neighbor_chunk_ids", []):
            if nid not in seen and state["status_by_id"].get(nid) == "NOT_STARTED":
                frontier.append(nid)
    rest = sorted([c for c in pending if c["id"] not in seen], key=lambda c: c["id"])
    return ordered + rest


def fetch_raw(bounds: dict) -> tuple[np.ndarray, str]:
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    z0, z1 = bounds["z"]
    sx, sy, sz = x1 - x0, y1 - y0, z1 - z0
    url = BASE + f"{sx}_{sy}_{sz}/{x0}_{y0}_{z0}"
    expected = sx * sy * sz
    with urlopen(url, timeout=600) as response:
        payload = response.read()
    if len(payload) != expected:
        raise RuntimeError(f"bad payload {len(payload)} != {expected}")
    return np.frombuffer(payload, dtype=np.uint8).reshape((sz, sy, sx)), url


def tile_layout(shape):
    counts = tuple(max(1, (n - c) // s + 2) for n, c, s in zip(shape, CROP_SIZE, STRIDE))
    padded_shape = tuple(c + (count - 1) * s for c, count, s in zip(CROP_SIZE, counts, STRIDE))
    padding = tuple(((p - n) // 2, (p - n + 1) // 2) for n, p in zip(shape, padded_shape))
    starts = tuple(tuple(i * step for i in range(count)) for count, step in zip(counts, STRIDE))
    return padding, starts


def gaussian_weight():
    zz, yy, xx = np.meshgrid(*(np.linspace(-1, 1, n, dtype=np.float32) for n in CROP_SIZE), indexing="ij")
    distance = np.sqrt(zz * zz + yy * yy + xx * xx)
    return 1e-6 + np.exp(-(distance**2 / (2.0 * 0.2**2)))


def infer_volume_batched(
    model,
    volume: np.ndarray,
    device: str,
    batch_size: int = BATCH,
    backend_meta: dict | None = None,
):
    import torch

    # Low-RAM hosts (Vast ~8GiB): float16 accum and/or disk memmap via env.
    accum_name = (os.environ.get("S7_ACCUM_DTYPE") or "float32").strip().lower()
    accum_dtype = np.float16 if accum_name in {"float16", "fp16", "half"} else np.float32
    memmap_root = (os.environ.get("S7_ACCUM_MEMMAP_DIR") or "").strip()
    progress_every = int(os.environ.get("S7_PROGRESS_EVERY", "256"))

    padding, starts = tile_layout(volume.shape)
    padded = np.pad(volume, padding, mode="reflect")
    sums_path = weights_path = None
    if memmap_root:
        mm_dir = Path(memmap_root)
        mm_dir.mkdir(parents=True, exist_ok=True)
        sums_path = mm_dir / f"sums_{os.getpid()}.dat"
        weights_path = mm_dir / f"weights_{os.getpid()}.dat"
        sums = np.memmap(sums_path, dtype=accum_dtype, mode="w+", shape=(4,) + padded.shape)
        weights = np.memmap(weights_path, dtype=accum_dtype, mode="w+", shape=padded.shape)
        sums[:] = 0
        weights[:] = 0
    else:
        sums = np.zeros((4,) + padded.shape, dtype=accum_dtype)
        weights = np.zeros(padded.shape, dtype=accum_dtype)
    patch_weight = gaussian_weight().astype(accum_dtype, copy=False)
    positions = list(product(*starts))
    total = len(positions)
    model = model.to(device).eval()
    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    use_amp = device.startswith("cuda")
    # MIOpen/cuDNN autotune for stable 20×64×64 tile shapes.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    print(
        f"    accum_dtype={accum_dtype.__name__} memmap={bool(memmap_root)} "
        f"tiles={total} progress_every={progress_every}",
        flush=True,
    )
    try:
        with torch.inference_mode():
            for start in range(0, total, batch_size):
                batch_pos = positions[start : start + batch_size]
                patches = np.empty((len(batch_pos), 1, *CROP_SIZE), dtype=np.float32)
                regions = []
                for i, position in enumerate(batch_pos):
                    region = tuple(slice(s, s + size) for s, size in zip(position, CROP_SIZE))
                    patches[i, 0] = np.ascontiguousarray(padded[region], dtype=np.float32) / 255.0
                    regions.append(region)
                host = torch.from_numpy(patches)
                # Direct H2D — pin_memory/non_blocking regressed throughput on ROCm 7.2 WSL.
                tensor = host.to(device)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        affinities, boundaries = model(tensor)
                else:
                    affinities, boundaries = model(tensor)
                prediction = torch.cat((affinities, boundaries), dim=1).float().cpu().numpy()
                if accum_dtype == np.float16:
                    prediction = prediction.astype(np.float16, copy=False)
                for i, region in enumerate(regions):
                    sums[(slice(None),) + region] += prediction[i] * patch_weight
                    weights[region] += patch_weight
                done = min(start + batch_size, total)
                if done % progress_every < batch_size or done == total:
                    elapsed = time.perf_counter() - t0
                    tps = done / elapsed if elapsed > 0 else float("nan")
                    print(f"    infer {done}/{total} tiles_per_sec={tps:.2f} batch={batch_size}", flush=True)
        infer_seconds = time.perf_counter() - t0
        original = tuple(slice(pad[0], pad[0] + size) for pad, size in zip(padding, volume.shape))
        try:
            del padded
        except Exception:  # noqa: BLE001
            pass

        out_aff = (os.environ.get("S7_STREAM_AFF_PATH") or "").strip()
        out_bnd = (os.environ.get("S7_STREAM_BND_PATH") or "").strip()
        stream_save = bool(out_aff and out_bnd)
        affinities = None
        boundaries = None
        spill_dir = None

        if stream_save:
            # Free GPU before host spill (critical on ~8GiB Vast hosts).
            try:
                import gc

                model.cpu()
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception:  # noqa: BLE001
                pass
            # Spill float16 cores in Z-slabs (~16MiB peak) — full-channel copies OOM when
            # the padded memmap is hot in page cache after a long infer.
            stage = Path(os.environ.get("S7_STAGE_DIR") or "/tmp/s7-fast-stage")
            spill = stage / f"spill_{os.getpid()}_{int(time.time())}"
            spill.mkdir(parents=True, exist_ok=True)
            z0s, y0s, x0s = (s.start for s in original)
            zz, yy, xx = volume.shape
            slab_z = int(os.environ.get("S7_SPILL_SLAB_Z", "8"))
            meta = {"shape": [zz, yy, xx], "slab_z": slab_z, "dtype": "float16"}
            (spill / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

            def _spill_volume(src, out_name: str, chan: int | None = None) -> None:
                with open(spill / out_name, "wb") as f:
                    for z0 in range(0, zz, slab_z):
                        z1 = min(z0 + slab_z, zz)
                        if chan is None:
                            slab = np.asarray(
                                src[z0s + z0 : z0s + z1, y0s : y0s + yy, x0s : x0s + xx],
                                dtype=np.float16,
                            )
                        else:
                            slab = np.asarray(
                                src[chan, z0s + z0 : z0s + z1, y0s : y0s + yy, x0s : x0s + xx],
                                dtype=np.float16,
                            )
                        f.write(np.ascontiguousarray(slab).tobytes())
                        del slab

            _spill_volume(weights, "w.raw", chan=None)
            try:
                del weights
            except Exception:  # noqa: BLE001
                pass
            if weights_path is not None and Path(weights_path).exists():
                try:
                    Path(weights_path).unlink()
                except OSError:
                    pass
                weights_path = None
            for c in range(4):
                _spill_volume(sums, f"c{c}.raw", chan=c)
            try:
                del sums
            except Exception:  # noqa: BLE001
                pass
            if sums_path is not None and Path(sums_path).exists():
                try:
                    Path(sums_path).unlink()
                except OSError:
                    pass
                sums_path = None
            spill_dir = str(spill)
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        else:
            core_w = np.array(weights[original], dtype=np.float32, copy=True)
            try:
                del weights
            except Exception:  # noqa: BLE001
                pass
            if weights_path is not None and Path(weights_path).exists():
                try:
                    Path(weights_path).unlink()
                except OSError:
                    pass
                weights_path = None
            affinities = np.empty((3,) + volume.shape, dtype=np.float32)
            for c in range(3):
                ch = np.array(sums[(c,) + original], dtype=np.float32, copy=True)
                ch /= core_w
                np.clip(ch, 0.0, 1.0, out=ch)
                affinities[c] = ch
                del ch
            boundaries = np.array(sums[(3,) + original], dtype=np.float32, copy=True)
            boundaries /= core_w
            np.clip(boundaries, 0.0, 1.0, out=boundaries)
            del core_w
            try:
                del sums
            except Exception:  # noqa: BLE001
                pass
            if sums_path is not None and Path(sums_path).exists():
                try:
                    Path(sums_path).unlink()
                except OSError:
                    pass
                sums_path = None
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()

        telemetry = {
            "tiles_total": total,
            "infer_seconds": infer_seconds,
            "tiles_per_second": total / infer_seconds if infer_seconds > 0 else None,
            "tile_batch_size": batch_size,
            "stride_zyx": list(STRIDE),
            "crop_size_zyx": list(CROP_SIZE),
            "amp": use_amp,
            "accum_dtype": accum_dtype.__name__,
            "accum_memmap": bool(memmap_root),
            "stream_save": stream_save,
            "spill_dir": spill_dir,
            "vram_peak_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "read_shape_zyx": list(volume.shape),
            "mode": "FAST_002_OPT",
            "pinned_h2d": False,
            "cudnn_benchmark": True,
            "infer_backend": (backend_meta or {}).get("applied", "eager"),
            "infer_backend_meta": backend_meta or {"applied": "eager"},
        }
        return affinities, boundaries, telemetry
    finally:
        if sums_path is not None or weights_path is not None:
            try:
                del sums
            except Exception:  # noqa: BLE001
                pass
            try:
                del weights
            except Exception:  # noqa: BLE001
                pass
            for p in (sums_path, weights_path):
                try:
                    if p is not None and Path(p).exists():
                        Path(p).unlink()
                except OSError:
                    pass


def process_chunk_fast(
    chunk: dict,
    model,
    device: str,
    batch_size: int = BATCH,
    prefetched: tuple[np.ndarray, str] | None = None,
    backend_meta: dict | None = None,
) -> dict:
    cid = chunk["id"]
    chunk_dir = OUT / "chunks" / cid
    if (chunk_dir / "affinities_core_czyx.npy").exists():
        raise FileExistsError(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    phases: dict[str, float] = {}
    wall0 = time.perf_counter()
    core = chunk["core_bounds_xyz"]
    print(f"  FAST fetch {cid} core={core}", flush=True)
    t = time.perf_counter()
    if prefetched is not None:
        raw, url = prefetched
        phases["dvid_fetch_seconds"] = 0.0
        phases["dvid_prefetch_hit"] = True
    else:
        raw, url = fetch_raw(core)
        phases["dvid_fetch_seconds"] = time.perf_counter() - t
        phases["dvid_prefetch_hit"] = False
    phases["raw_save_seconds"] = 0.0  # T5: skip
    print(f"  FAST infer {cid} shape={raw.shape}", flush=True)
    aff_dst = chunk_dir / "affinities_core_czyx.npy"
    bnd_dst = chunk_dir / "boundaries_core.tif"
    use_stream = bool((os.environ.get("S7_ACCUM_MEMMAP_DIR") or "").strip()) or (
        (os.environ.get("S7_STREAM_SAVE") or "").strip().lower() in {"1", "true", "yes"}
    )
    if use_stream:
        os.environ["S7_STREAM_AFF_PATH"] = str(aff_dst)
        os.environ["S7_STREAM_BND_PATH"] = str(bnd_dst)
    affinities, boundaries, infer_tel = infer_volume_batched(
        model, raw, device=device, batch_size=batch_size, backend_meta=backend_meta
    )
    del raw
    for _k in ("S7_STREAM_AFF_PATH", "S7_STREAM_BND_PATH"):
        os.environ.pop(_k, None)
    phases["affinity_infer_seconds"] = float(infer_tel["infer_seconds"])
    t = time.perf_counter()
    import gc
    import tifffile
    from numpy.lib.format import write_array_header_1_0

    spill_dir = infer_tel.get("spill_dir")
    if spill_dir:
        # Ensure model is off GPU; assemble float32 one Z-slab at a time.
        try:
            import torch

            model.cpu()
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:  # noqa: BLE001
            pass
        spill = Path(spill_dir)
        meta = json.loads((spill / "meta.json").read_text(encoding="utf-8"))
        zz, yy, xx = meta["shape"]
        slab_z = int(meta.get("slab_z", 8))
        with open(aff_dst, "wb") as f:
            write_array_header_1_0(
                f,
                {
                    "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
                    "fortran_order": False,
                    "shape": (3, zz, yy, xx),
                },
            )
            for c in range(3):
                with open(spill / f"c{c}.raw", "rb") as cf, open(spill / "w.raw", "rb") as wf:
                    for z0 in range(0, zz, slab_z):
                        z1 = min(z0 + slab_z, zz)
                        n = (z1 - z0) * yy * xx
                        ch = np.frombuffer(cf.read(n * 2), dtype=np.float16).astype(np.float32)
                        w = np.frombuffer(wf.read(n * 2), dtype=np.float16).astype(np.float32)
                        ch /= w
                        np.clip(ch, 0.0, 1.0, out=ch)
                        f.write(np.ascontiguousarray(ch).tobytes())
                        del ch, w
                (spill / f"c{c}.raw").unlink(missing_ok=True)
        # Boundaries: keep float16 host buffer (~0.25GiB) then write TIFF.
        bnd = np.empty((zz, yy, xx), dtype=np.float16)
        with open(spill / "c3.raw", "rb") as cf, open(spill / "w.raw", "rb") as wf:
            for z0 in range(0, zz, slab_z):
                z1 = min(z0 + slab_z, zz)
                n = (z1 - z0) * yy * xx
                ch = np.frombuffer(cf.read(n * 2), dtype=np.float16).astype(np.float32)
                w = np.frombuffer(wf.read(n * 2), dtype=np.float16).astype(np.float32)
                ch /= w
                np.clip(ch, 0.0, 1.0, out=ch)
                bnd[z0:z1] = ch.reshape((z1 - z0, yy, xx)).astype(np.float16)
                del ch, w
        (spill / "c3.raw").unlink(missing_ok=True)
        (spill / "w.raw").unlink(missing_ok=True)
        tifffile.imwrite(bnd_dst, bnd)
        del bnd
        import shutil

        shutil.rmtree(spill, ignore_errors=True)
        try:
            model.cuda()
        except Exception:  # noqa: BLE001
            pass
        infer_tel["stream_save"] = True
    else:
        direct = (os.environ.get("S7_DIRECT_SAVE") or "1").strip() not in {"0", "false", "no"}
        if direct:
            np.save(aff_dst, affinities, allow_pickle=False)
            tifffile.imwrite(bnd_dst, boundaries)
        else:
            import shutil

            STAGE_DIR.mkdir(parents=True, exist_ok=True)
            stage_dir = STAGE_DIR / cid
            if stage_dir.exists():
                shutil.rmtree(stage_dir, ignore_errors=True)
            stage_dir.mkdir(parents=True, exist_ok=True)
            aff_stage = stage_dir / "affinities_core_czyx.npy"
            bnd_stage = stage_dir / "boundaries_core.tif"
            np.save(aff_stage, affinities, allow_pickle=False)
            tifffile.imwrite(bnd_stage, boundaries)
            for src, dst in ((aff_stage, aff_dst), (bnd_stage, bnd_dst)):
                try:
                    src.replace(dst)
                except OSError:
                    shutil.copy2(src, dst)
                    src.unlink(missing_ok=True)
            shutil.rmtree(stage_dir, ignore_errors=True)
        if affinities is not None:
            del affinities
        if boundaries is not None:
            del boundaries
    phases["core_crop_save_seconds"] = time.perf_counter() - t
    phases["segmentation_seconds"] = 0.0
    phases["total_wall_seconds"] = time.perf_counter() - wall0
    status = "AFFINITY_DONE_SEG_PENDING"
    output_bytes = {p.name: p.stat().st_size for p in chunk_dir.iterdir() if p.is_file()}
    aff_shape = list(np.load(aff_dst, mmap_mode="r").shape)
    receipt = {
        "chunk_id": cid,
        "created_at": _now(),
        "throughput_contract_id": "AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_001",
        "source_url": url,
        "core_bounds_xyz": core,
        "read_bounds_xyz": core,  # core-only
        "affinities_shape_czyx": aff_shape,
        "segmentation": {"deferred": True},
        "status": status,
        "classification": "MACHINE_PSEUDOLABEL_CHUNK",
        "checkpoint_sha256": sha256(win_to_wsl(CKPT)),
        "throughput_telemetry": {
            **infer_tel,
            "phase_seconds": phases,
            "output_bytes": output_bytes,
            "naive_serial_full_volume_gpu_hours": N_CHUNKS * phases["affinity_infer_seconds"] / 3600.0,
            "stage_dir": str(STAGE_DIR),
        },
    }
    (chunk_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (chunk_dir / "throughput_telemetry.json").write_text(
        json.dumps(receipt["throughput_telemetry"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # AWS S3 artifact upload removed — Affinity production cloud is Vast.ai.
    # Optional pull/push of chunk dirs is handled outside this worker (rsync/scp).
    if os.environ.get("S7_ARTIFACT_BUCKET", "").strip():
        print(
            f"  warning {cid}: S7_ARTIFACT_BUCKET is ignored (AWS Affinity path decommissioned)",
            flush=True,
        )
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=4)
    ap.add_argument("--loop", action="store_true", help="Keep taking batches until NOT_STARTED is empty")
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--claim", action="store_true", help="Use exclusive claim-lock (FAST_002 multi-worker)")
    ap.add_argument("--contract", default="FAST_002")
    ap.add_argument("--no-prefetch", action="store_true", help="Disable DVID prefetch of next chunk")
    ap.add_argument(
        "--infer-backend",
        default=os.environ.get("S7_INFER_BACKEND", "eager"),
        choices=["eager", "compile", "migraphx", "tensorrt"],
        help="Inference accelerator: eager | compile (torch.compile) | migraphx | tensorrt",
    )
    ap.add_argument(
        "--shard",
        default=os.environ.get("S7_SHARD", ""),
        help="Split work with no shared server: e.g. 0/2 local, 1/2 Vast (no overlapping chunks)",
    )
    args = ap.parse_args()
    batch_size = int(args.batch_size)
    shard = None
    if str(args.shard).strip():
        a, b = str(args.shard).strip().split("/", 1)
        shard = (int(a), int(b))
        if not (0 <= shard[0] < shard[1]):
            raise SystemExit(f"bad --shard {args.shard}")

    if json.loads(S6.read_text(encoding="utf-8")).get("status") != "APPROVED":
        raise SystemExit("S6 not approved")
    contract_path = CONTRACT_FAST_002 if args.contract in {"FAST_002", "FAST_003"} else CONTRACT_FAST_001
    if args.contract == "FAST_003":
        contract_path = REPO / "experiments/phase6e/AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_003.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.exists() else {"id": args.contract}
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))["chunks"]
    roi = roi_bounds_xyz(json.loads(ROI_PKG.read_text(encoding="utf-8")))

    OUT.mkdir(parents=True, exist_ok=True)
    state = load_state(chunks)
    state["fast_contract_id"] = contract.get("id", args.contract)
    state["opened_at"] = state.get("opened_at") or _now()
    save_state(state)

    import torch
    from concurrent.futures import ThreadPoolExecutor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm required")
    sys.path.insert(0, str(REPO / "third_party/segneuron/Train_and_Inference"))
    from model.Mnet import MNet
    from s7_infer_accelerate import accelerate_model

    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub")
    ckpt = torch.load(win_to_wsl(CKPT), map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_weights"], strict=True)
    model, backend_meta = accelerate_model(model.cuda(), args.infer_backend, sample_batch=batch_size)
    print(f"infer_backend={json.dumps(backend_meta)}", flush=True)

    from s7_chunk_claim import (  # noqa: E402
        claim_backend,
        release_local,
        try_claim_local,
        worker_id,
    )
    from s7_durable_commit import durable_commit  # noqa: E402

    wid = worker_id()
    backend = claim_backend()
    claim_http = os.environ.get("S7_CLAIM_HTTP", "").strip()
    use_http = backend == "http" or bool(claim_http)
    print(
        f"worker_id={wid} claim={args.claim} backend={backend} contract={args.contract} "
        f"batch={batch_size} prefetch={not args.no_prefetch} infer={args.infer_backend} "
        f"claim_http={claim_http or None} durable={os.environ.get('S7_DURABLE_MODE', 'local')} "
        f"shard={args.shard or 'all'}",
        flush=True,
    )

    # Optional fleet registration / telemetry
    if use_http and claim_http:
        try:
            from cloud.claim_http import register_worker_http, telemetry_http

            register_worker_http(
                claim_http,
                {
                    "worker_id": wid,
                    "gpu": torch.cuda.get_device_name(0),
                    "runtime": "CUDA" if "rocm" not in (getattr(torch.version, "hip", None) or "") else "ROCm",
                    "batch": batch_size,
                    "backend": args.infer_backend,
                    "dph_usd": float(os.environ.get("S7_VAST_DPH", "0") or 0),
                    "instance_id": os.environ.get("S7_VAST_INSTANCE_ID", ""),
                },
            )
        except Exception as reg_exc:  # noqa: BLE001
            print(f"  register-worker warning: {reg_exc}", flush=True)

    chunks_by_id = {c["id"]: c for c in chunks}
    total_done_run = []
    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    while True:
        if use_http and args.claim:
            # Shared coordinator assigns work — do not trust a local queue_state copy.
            from cloud.claim_http import next_claim_http, telemetry_http

            url = claim_http or os.environ.get("S7_CLAIM_HTTP", "")
            batch_ids = []
            for _ in range(args.max_chunks):
                cid = next_claim_http(url, wid=wid)
                if not cid:
                    break
                batch_ids.append(cid)
            print(f"FAST http claimed={len(batch_ids)} batch={batch_size}", flush=True)
            if not batch_ids:
                break
            batch_chunks = [chunks_by_id[cid] for cid in batch_ids if cid in chunks_by_id]
            if len(batch_chunks) != len(batch_ids):
                missing = set(batch_ids) - set(chunks_by_id)
                print(f"  missing chunk defs: {missing}", flush=True)
        else:
            state = load_state(chunks)
            queue = priority_queue(chunks, state, roi, shard=shard)
            save_state(state)
            print(
                f"FAST queue pending={len(queue)} max={args.max_chunks} batch={batch_size} shard={args.shard or 'all'}",
                flush=True,
            )
            if not queue:
                break
            batch_chunks = queue[: args.max_chunks]

        done = []
        prefetch_future = None
        prefetched_raw = None
        for idx, chunk in enumerate(batch_chunks):
            cid = chunk["id"]
            if args.claim and not use_http:
                if not try_claim_local(cid, wid=wid):
                    print(f"  skip {cid} (claim lost)", flush=True)
                    continue
            elif not args.claim:
                state = load_state(chunks)
                if state["status_by_id"].get(cid) == "RUNNING":
                    aff = OUT / "chunks" / cid / "affinities_core_czyx.npy"
                    if aff.exists():
                        print(f"  skip {cid} (RUNNING with affinities)", flush=True)
                        continue
                state["status_by_id"][cid] = "RUNNING"
                save_state(state)
            try:
                use_prefetch = None
                if prefetched_raw is not None:
                    use_prefetch = prefetched_raw
                    prefetched_raw = None
                elif prefetch_future is not None:
                    try:
                        use_prefetch = prefetch_future.result()
                    except Exception as pref_exc:  # noqa: BLE001
                        print(f"  prefetch miss: {pref_exc}", flush=True)
                        use_prefetch = None
                    prefetch_future = None

                if not args.no_prefetch and idx + 1 < len(batch_chunks):
                    nxt = batch_chunks[idx + 1]
                    prefetch_future = prefetch_pool.submit(fetch_raw, nxt["core_bounds_xyz"])

                receipt = process_chunk_fast(
                    chunk,
                    model,
                    device="cuda",
                    batch_size=batch_size,
                    prefetched=use_prefetch,
                    backend_meta=backend_meta,
                )
                receipt["throughput_contract_id"] = (
                    "AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_002"
                    if args.contract == "FAST_002"
                    else "AFFINITY_S7_THROUGHPUT_CONTRACT_FAST_001"
                )
                receipt["worker_id"] = wid
                (OUT / "chunks" / cid / "receipt.json").write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                if args.claim:
                    commit_meta = durable_commit(cid, final_status=receipt["status"])
                    receipt["durable_commit"] = commit_meta
                else:
                    state = load_state(chunks)
                    state["status_by_id"][cid] = receipt["status"]
                    state.setdefault("completed", []).append(cid)
                    save_state(state)
                done.append(receipt)
                total_done_run.append(cid)
                tel = receipt["throughput_telemetry"]
                tps = tel.get("tiles_per_second")
                if use_http and claim_http:
                    try:
                        from cloud.claim_http import telemetry_http

                        telemetry_http(
                            claim_http,
                            {
                                "worker_id": wid,
                                "status": "ACTIVE",
                                "tiles_per_second": tps,
                                "last_chunk": cid,
                                "chunks_this_run": len(total_done_run),
                            },
                        )
                    except Exception:
                        pass
                print(
                    json.dumps(
                        {
                            "chunk": cid,
                            "tiles": tel["tiles_total"],
                            "tiles_per_sec": tel["tiles_per_second"],
                            "infer_s": tel["infer_seconds"],
                            "naive_serial_gpu_days": tel["naive_serial_full_volume_gpu_hours"] / 24.0,
                            "worker_id": wid,
                            "batch": batch_size,
                            "prefetch_hit": tel.get("phase_seconds", {}).get("dvid_prefetch_hit"),
                            "persist_s": (receipt.get("durable_commit") or {}).get("persist_seconds"),
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                if args.claim:
                    if use_http and claim_http:
                        from cloud.claim_http import complete_http

                        try:
                            complete_http(cid, claim_http, "FAILED")
                        except Exception:
                            release_local(cid, "FAILED")
                    else:
                        release_local(cid, "FAILED")
                else:
                    state = load_state(chunks)
                    state["status_by_id"][cid] = "FAILED"
                    state.setdefault("failed", []).append({"id": cid, "error": str(exc)})
                    save_state(state)
                print(f"FAILED {cid}: {exc}", flush=True)

        # Progress rollup from coordinator when http, else local
        if use_http and claim_http:
            try:
                from urllib.request import urlopen

                st = json.loads(urlopen(claim_http.rstrip("/") + "/status", timeout=30).read().decode())
                counts = st.get("counts") or {}
                n_done = counts.get("COMPLETE", 0) + counts.get("AFFINITY_DONE_SEG_PENDING", 0)
                n_left = counts.get("NOT_STARTED", 0)
            except Exception:
                n_done, n_left = len(total_done_run), -1
        else:
            state = load_state(chunks)
            n_done = sum(1 for s in state["status_by_id"].values() if s in {"COMPLETE", "AFFINITY_DONE_SEG_PENDING"})
            n_left = sum(1 for s in state["status_by_id"].values() if s == "NOT_STARTED")
        summary = {
            "id": "AFFINITY_FULLVOL_S7_001_PROGRESS",
            "updated_at": _now(),
            "mode": "FLEET_HTTP" if use_http else ("FAST_002" if args.claim else "FAST_001"),
            "chunks_completed_this_run": total_done_run[-50:],
            "n_done": n_done,
            "n_not_started": n_left,
            "n_total": len(chunks),
            "status": "RUNNING" if n_left else "COMPLETE",
            "tile_batch_size": batch_size,
            "worker_id": wid,
        }
        (OUT / "PROGRESS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        if not args.loop or n_left == 0:
            break
    prefetch_pool.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
