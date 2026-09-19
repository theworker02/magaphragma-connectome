"""Production tile packing — same tile math, larger GPU packs + overlapped prep."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Any

import numpy as np

from hyperdrain.geometry import CROP_ZYX, STRIDE_ZYX, tile_layout
from hyperdrain.pipeline import gaussian_weight


def production_tile_positions(
    volume_zyx: tuple[int, int, int],
    crop_zyx: tuple[int, int, int] = CROP_ZYX,
    stride_zyx: tuple[int, int, int] = STRIDE_ZYX,
) -> list[tuple[int, int, int]]:
    """Immutable production tile MAP (same starts as frozen eager tiled)."""
    layout = tile_layout(volume_zyx, crop_zyx, stride_zyx)
    starts = [tuple(i * step for i in range(count)) for count, step in zip(layout.counts_zyx, stride_zyx)]
    # Locality-friendly order: Z slowest already via product(z,y,x) — keep production map order.
    return list(product(*starts))


def order_tiles_for_locality(positions: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Reorder execution only; map identity unchanged. Prefer Z-major spatial locality."""
    return sorted(positions, key=lambda p: (p[0], p[1], p[2]))


class PackBuffers:
    """Preallocated host/device tensors for steady-state packed inference."""

    def __init__(self, pack_size: int, crop_zyx: tuple[int, int, int] = CROP_ZYX, device: str = "cuda"):
        import torch

        self.pack_size = pack_size
        self.crop_zyx = crop_zyx
        self.device = device
        self.host_in = torch.empty((pack_size, 1, *crop_zyx), dtype=torch.float32, pin_memory=True)
        self.gpu_in = torch.empty((pack_size, 1, *crop_zyx), dtype=torch.float32, device=device)
        # MNet: affinities (B,3,Z,Y,X) + boundaries (B,1,Z,Y,X)
        self.gpu_out = torch.empty((pack_size, 4, *crop_zyx), dtype=torch.float32, device=device)
        self.host_out = torch.empty((pack_size, 4, *crop_zyx), dtype=torch.float32, pin_memory=True)

    def fill_host(self, patches: np.ndarray) -> int:
        import torch

        n = int(patches.shape[0])
        self.host_in[:n].copy_(torch.from_numpy(patches))
        return n


def _extract_pack(
    padded: np.ndarray,
    positions: list[tuple[int, int, int]],
    crop_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, list[tuple]]:
    n = len(positions)
    patches = np.empty((n, 1, *crop_zyx), dtype=np.float32)
    regions = []
    for i, position in enumerate(positions):
        region = tuple(slice(s, s + size) for s, size in zip(position, crop_zyx))
        patches[i, 0] = np.ascontiguousarray(padded[region], dtype=np.float32) / 255.0
        regions.append(region)
    return patches, regions


def infer_volume_packed(
    model: Any,
    volume: np.ndarray,
    *,
    pack_size: int = 16,
    crop_zyx: tuple[int, int, int] = CROP_ZYX,
    stride_zyx: tuple[int, int, int] = STRIDE_ZYX,
    device: str = "cuda",
    use_amp: bool = True,
    prefetch: bool = True,
    locality_order: bool = True,
    progress_path: Any | None = None,
    resume_from: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Packed production inference.

    Each packed element receives the exact input tensor eager tiled would supply
    for that production tile_id. Packing changes scheduling only.
    """
    import torch

    layout = tile_layout(tuple(volume.shape), crop_zyx, stride_zyx)
    padded = np.pad(volume, layout.padding, mode="reflect")
    sums = np.zeros((4,) + layout.padded_zyx, dtype=np.float32)
    weights = np.zeros(layout.padded_zyx, dtype=np.float32)
    patch_weight = gaussian_weight(crop_zyx)

    positions = production_tile_positions(tuple(volume.shape), crop_zyx, stride_zyx)
    if locality_order:
        positions = order_tiles_for_locality(positions)
    total = len(positions)
    if resume_from > 0:
        positions = positions[resume_from:]
    done0 = resume_from

    model = model.to(device).eval()
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats()

    buf_a = PackBuffers(pack_size, crop_zyx, device)
    buf_b = PackBuffers(pack_size, crop_zyx, device) if prefetch else None
    pool = ThreadPoolExecutor(max_workers=1) if prefetch else None

    def run_pack(buf: PackBuffers, patches: np.ndarray, regions: list[tuple]) -> None:
        n = buf.fill_host(patches)
        buf.gpu_in[:n].copy_(buf.host_in[:n], non_blocking=True)
        with torch.inference_mode():
            if use_amp and device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    affinities, boundaries = model(buf.gpu_in[:n])
            else:
                affinities, boundaries = model(buf.gpu_in[:n])
            pred = torch.cat((affinities, boundaries), dim=1).float()
            buf.gpu_out[:n].copy_(pred, non_blocking=True)
            buf.host_out[:n].copy_(buf.gpu_out[:n], non_blocking=True)
        torch.cuda.synchronize()
        out = buf.host_out[:n].numpy()
        for i, region in enumerate(regions):
            sums[(slice(None),) + region] += out[i] * patch_weight
            weights[region] += patch_weight

    t0 = time.perf_counter()
    packs = [positions[i : i + pack_size] for i in range(0, len(positions), pack_size)]
    future = None
    next_payload = None

    def prepare(pack_pos: list[tuple[int, int, int]]):
        return _extract_pack(padded, pack_pos, crop_zyx)

    for pi, pack_pos in enumerate(packs):
        if prefetch and pool is not None:
            if future is None:
                patches, regions = prepare(pack_pos)
            else:
                patches, regions = future.result()
            if pi + 1 < len(packs):
                future = pool.submit(prepare, packs[pi + 1])
            else:
                future = None
            buf = buf_a if (pi % 2 == 0) else (buf_b or buf_a)
        else:
            patches, regions = prepare(pack_pos)
            buf = buf_a
        run_pack(buf, patches, regions)
        done = done0 + min((pi + 1) * pack_size, len(positions))
        if progress_path is not None and (done % max(pack_size * 4, 64) < pack_size or done >= total):
            try:
                progress_path.write_text(
                    __import__("json").dumps({"tiles_done": done, "tiles_total": total, "pack_size": pack_size}) + "\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
        if done % 256 < pack_size or done == total:
            elapsed = time.perf_counter() - t0
            tps = (done - done0) / elapsed if elapsed > 0 else float("nan")
            print(f"    hyperdrain packed {done}/{total} tiles_per_sec={tps:.2f} pack={pack_size}", flush=True)

    if pool is not None:
        pool.shutdown(wait=False)

    infer_s = time.perf_counter() - t0
    # Avoid div0 on resume-empty
    np.maximum(weights, 1e-12, out=weights)
    sums /= weights[None]
    np.clip(sums, 0.0, 1.0, out=sums)
    original = tuple(slice(pad[0], pad[0] + size) for pad, size in zip(layout.padding, volume.shape))
    tel = {
        "mode": "packed",
        "backend_mode": "hyperdrain:packed",
        "tiles_total": total,
        "tiles_resumed_from": resume_from,
        "infer_seconds": infer_s,
        "tiles_per_second": (total - resume_from) / infer_s if infer_s > 0 else None,
        "pack_size": pack_size,
        "tile_batch_size": pack_size,
        "crop_size_zyx": list(crop_zyx),
        "stride_zyx": list(stride_zyx),
        "amp": use_amp,
        "prefetch": prefetch,
        "locality_order": locality_order,
        "vram_peak_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "read_shape_zyx": list(volume.shape),
        "production_tile_map_n": total,
    }
    return sums[(slice(0, 3),) + original].copy(), sums[(3,) + original].copy(), tel
