"""Compatibility spatial API for Connectome tools (wraps public _core.spatial)."""
from __future__ import annotations

from typing import Iterator

import numpy as np

from .spatial import (  # noqa: F401
    AABB as _AABB,
    TileIndex as _TileIndex,
    morton_key,
    overlap_volume,
    tile_to_aabb,
    world_to_tile,
)
from .spatial import *  # noqa: F403


def morton3(z: int, y: int, x: int) -> int:
    return morton_key(z, y, x)


class AABB(_AABB):
    def as_dict(self) -> dict:
        return self.to_dict()

    @property
    def volume_voxels(self) -> int:
        return self.volume()


class TileIndex(_TileIndex):
    """Tool-facing TileIndex: iter_tiles yields (key, AABB), neighbors take keys."""

    def __init__(self, shape_zyx, tile_zyx=(32, 64, 64), **kwargs):
        super().__init__(shape_zyx, tile_zyx, **kwargs)
        self.shape_zyx = tuple(int(x) for x in shape_zyx)
        self.tile_zyx = tuple(int(x) for x in tile_zyx)

    @staticmethod
    def _key(iz: int, iy: int, ix: int) -> str:
        return f"T-Z{iz:03d}-Y{iy:03d}-X{ix:03d}"

    @staticmethod
    def _parse(key: str) -> tuple[int, int, int]:
        parts = key.replace("T-", "").split("-")
        return tuple(int(p[1:]) for p in parts)  # type: ignore[return-value]

    def iter_tiles(self) -> Iterator[tuple[str, AABB]]:
        for iz, iy, ix in _TileIndex.iter_tiles(self):
            box = self.aabb_of(iz, iy, ix)
            yield self._key(iz, iy, ix), AABB(
                box.z0, box.y0, box.x0, box.z1, box.y1, box.x1
            )

    def neighbors(self, key: str, halo: int = 1) -> list[str]:  # type: ignore[override]
        iz, iy, ix = self._parse(key)
        return [self._key(*c) for c in _TileIndex.neighbors(self, iz, iy, ix, halo=halo)]

    def crop(self, volume: np.ndarray, box: AABB) -> np.ndarray:
        return volume[box.z0 : box.z1, box.y0 : box.y1, box.x0 : box.x1]