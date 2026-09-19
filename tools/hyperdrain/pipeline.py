"""Inference pipelines: tiled (eager contract) + macro-dense + prefetch."""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import product
from typing import Any, Callable

import numpy as np

from hyperdrain.geometry import CROP_ZYX, STRIDE_ZYX, TileLayout, tile_layout


def gaussian_weight(crop_zyx: tuple[int, int, int] = CROP_ZYX) -> np.ndarray:
    zz, yy, xx = np.meshgrid(*(np.linspace(-1, 1, n, dtype=np.float32) for n in crop_zyx), indexing="ij")
    distance = np.sqrt(zz * zz + yy * yy + xx * xx)
    return 1e-6 + np.exp(-(distance**2 / (2.0 * 0.2**2)))


class BufferPool:
    """Persistent host/device buffers to cut allocator churn."""

    def __init__(self, batch: int, crop_zyx: tuple[int, int, int] = CROP_ZYX, device: str = "cuda"):
        import torch

        self.batch = batch
        self.crop_zyx = crop_zyx
        self.device = device
        self.host = torch.empty((batch, 1, *crop_zyx), dtype=torch.float32, pin_memory=device.startswith("cuda"))
        self.gpu = torch.empty((batch, 1, *crop_zyx), dtype=torch.float32, device=device)

    def upload(self, patches: np.ndarray) -> Any:
        import torch

        n = patches.shape[0]
        self.host[:n].copy_(torch.from_numpy(patches))
        self.gpu[:n].copy_(self.host[:n], non_blocking=True)
        return self.gpu[:n]


def infer_volume_tiled(
    model: Any,
    volume: np.ndarray,
    *,
    batch_size: int = 16,
    crop_zyx: tuple[int, int, int] = CROP_ZYX,
    stride_zyx: tuple[int, int, int] = STRIDE_ZYX,
    device: str = "cuda",
    use_amp: bool = True,
    use_pinned: bool = False,
    pool: BufferPool | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Bit-compatible stitch with FAST worker: reflect pad + gaussian blend."""
    import torch

    layout = tile_layout(tuple(volume.shape), crop_zyx, stride_zyx)
    padded = np.pad(volume, layout.padding, mode="reflect")
    sums = np.zeros((4,) + layout.padded_zyx, dtype=np.float32)
    weights = np.zeros(layout.padded_zyx, dtype=np.float32)
    patch_weight = gaussian_weight(crop_zyx)
    starts = [tuple(i * step for i in range(count)) for count, step in zip(layout.counts_zyx, stride_zyx)]
    positions = list(product(*starts))
    total = len(positions)
    model = model.to(device).eval()
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch_pos = positions[start : start + batch_size]
            n = len(batch_pos)
            patches = np.empty((n, 1, *crop_zyx), dtype=np.float32)
            regions = []
            for i, position in enumerate(batch_pos):
                region = tuple(slice(s, s + size) for s, size in zip(position, crop_zyx))
                patches[i, 0] = np.ascontiguousarray(padded[region], dtype=np.float32) / 255.0
                regions.append(region)
            if use_pinned and pool is not None and n == pool.batch:
                tensor = pool.upload(patches)
            else:
                tensor = torch.from_numpy(patches).to(device)
            if use_amp and device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    affinities, boundaries = model(tensor)
            else:
                affinities, boundaries = model(tensor)
            prediction = torch.cat((affinities, boundaries), dim=1).float().cpu().numpy()
            for i, region in enumerate(regions):
                sums[(slice(None),) + region] += prediction[i] * patch_weight
                weights[region] += patch_weight
            done = min(start + batch_size, total)
            if done % 256 < batch_size or done == total:
                elapsed = time.perf_counter() - t0
                tps = done / elapsed if elapsed > 0 else float("nan")
                print(f"    hyperdrain tiled {done}/{total} tiles_per_sec={tps:.2f} batch={batch_size}", flush=True)

    infer_s = time.perf_counter() - t0
    sums /= weights[None]
    np.clip(sums, 0.0, 1.0, out=sums)
    original = tuple(slice(pad[0], pad[0] + size) for pad, size in zip(layout.padding, volume.shape))
    tel = {
        "mode": "tiled",
        "tiles_total": total,
        "infer_seconds": infer_s,
        "tiles_per_second": total / infer_s if infer_s > 0 else None,
        "tile_batch_size": batch_size,
        "crop_size_zyx": list(crop_zyx),
        "stride_zyx": list(stride_zyx),
        "amp": use_amp,
        "pinned": use_pinned,
        "vram_peak_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "read_shape_zyx": list(volume.shape),
    }
    return sums[(slice(0, 3),) + original].copy(), sums[(3,) + original].copy(), tel


def infer_volume_macro(
    model: Any,
    volume: np.ndarray,
    *,
    macro_zyx: tuple[int, int, int],
    stride_zyx: tuple[int, int, int] | None = None,
    device: str = "cuda",
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Dense/macro-tile path: larger spatial forwards, same Gaussian blend contract.
    stride defaults to macro size (no overlap) except keep Z overlap if macro_z > 20.
    """
    import torch

    if stride_zyx is None:
        # Preserve Z overlap ratio of production (stride_z = macro_z/2) when macro_z even;
        # XY abut (stride = size) like production.
        mz, my, mx = macro_zyx
        stride_zyx = (max(1, mz // 2), my, mx)

    return infer_volume_tiled(
        model,
        volume,
        batch_size=1,
        crop_zyx=macro_zyx,
        stride_zyx=stride_zyx,
        device=device,
        use_amp=use_amp,
        use_pinned=False,
        pool=None,
    )


class PrefetchPipeline:
    """While GPU runs N, prefetch raw for N+1 on a side thread."""

    def __init__(self, fetch_fn: Callable[[dict], tuple[np.ndarray, str]], max_workers: int = 1):
        self.fetch_fn = fetch_fn
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self._future: Future | None = None

    def kick(self, bounds: dict | None) -> None:
        if bounds is None:
            return
        self._future = self.pool.submit(self.fetch_fn, bounds)

    def take(self) -> tuple[np.ndarray, str] | None:
        if self._future is None:
            return None
        try:
            out = self._future.result()
        except Exception as exc:  # noqa: BLE001
            print(f"  hyperdrain prefetch miss: {exc}", flush=True)
            out = None
        self._future = None
        return out

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False)
