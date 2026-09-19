"""Simulation backends: reference, halo-cache, cascade (experimental)."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class BackendResult:
    name: str
    checksum: str
    seconds: float
    cache_hit: bool = False
    output: np.ndarray | None = None


def _checksum(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def reference_infer(tile: np.ndarray) -> BackendResult:
    t0 = time.perf_counter()
    # Deterministic CPU "affinity-like" transform (simulation harness).
    out = (tile.astype(np.float32) / 255.0)
    out = np.clip(out * 0.91 + 0.03, 0.0, 1.0)
    dt = time.perf_counter() - t0
    return BackendResult("reference", _checksum(out), dt, output=out)


class HaloCache:
    """Accepted acceleration: reuse border-identical neighbor compute when safe."""

    def __init__(self) -> None:
        self.store: dict[str, BackendResult] = {}
        self.hits = 0
        self.misses = 0

    def key(self, tile: np.ndarray) -> str:
        # Border halo fingerprint (outer ring) — identical halo => cacheable.
        if tile.ndim != 3:
            return _checksum(tile)
        z, y, x = tile.shape
        border = np.concatenate(
            [
                tile[0, :, :].ravel(),
                tile[-1, :, :].ravel(),
                tile[:, 0, :].ravel(),
                tile[:, -1, :].ravel(),
            ]
        )
        return hashlib.sha256(border.tobytes()).hexdigest()[:24]

    def infer(self, tile: np.ndarray) -> BackendResult:
        k = self.key(tile)
        if k in self.store:
            self.hits += 1
            hit = self.store[k]
            return BackendResult("halo_cache", hit.checksum, 1e-6, cache_hit=True, output=hit.output)
        self.misses += 1
        ref = reference_infer(tile)
        self.store[k] = ref
        return BackendResult("halo_cache", ref.checksum, ref.seconds, cache_hit=False, output=ref.output)

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


def cascade_infer(tile: np.ndarray) -> BackendResult:
    """Experimental: coarser pass — intentionally may fail equivalence."""
    t0 = time.perf_counter()
    coarse = tile[::2, ::2, ::2].astype(np.float32) / 255.0
    # upsample-ish repeat
    out = np.repeat(np.repeat(np.repeat(coarse, 2, 0), 2, 1), 2, 2)
    z, y, x = tile.shape
    out = out[:z, :y, :x]
    dt = time.perf_counter() - t0
    return BackendResult("cascade", _checksum(out), dt, output=out)
