"""Spatial primitives: AABB, ZYX tiling, Morton keys, halo neighbors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

Coord3 = Tuple[int, int, int]
Float3 = Tuple[float, float, float]


def _as_int3(v: Sequence[int | float]) -> Coord3:
    if len(v) != 3:
        raise ValueError("expected length-3 ZYX coordinate")
    return (int(v[0]), int(v[1]), int(v[2]))


@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box in ZYX order (inclusive min, exclusive max)."""

    z0: int
    y0: int
    x0: int
    z1: int
    y1: int
    x1: int

    @classmethod
    def from_corners(cls, mn: Sequence[int], mx: Sequence[int]) -> "AABB":
        z0, y0, x0 = _as_int3(mn)
        z1, y1, x1 = _as_int3(mx)
        return cls(z0, y0, x0, z1, y1, x1)

    @classmethod
    def from_shape(cls, shape: Sequence[int], origin: Sequence[int] = (0, 0, 0)) -> "AABB":
        oz, oy, ox = _as_int3(origin)
        sz, sy, sx = _as_int3(shape)
        return cls(oz, oy, ox, oz + sz, oy + sy, ox + sx)

    @property
    def min_corner(self) -> Coord3:
        return (self.z0, self.y0, self.x0)

    @property
    def max_corner(self) -> Coord3:
        return (self.z1, self.y1, self.x1)

    @property
    def shape(self) -> Coord3:
        return (max(0, self.z1 - self.z0), max(0, self.y1 - self.y0), max(0, self.x1 - self.x0))

    def volume(self) -> int:
        sz, sy, sx = self.shape
        return int(sz) * int(sy) * int(sx)

    def valid(self) -> bool:
        return self.z1 > self.z0 and self.y1 > self.y0 and self.x1 > self.x0

    def intersects(self, other: "AABB") -> bool:
        return not (
            self.z1 <= other.z0
            or other.z1 <= self.z0
            or self.y1 <= other.y0
            or other.y1 <= self.y0
            or self.x1 <= other.x0
            or other.x1 <= self.x0
        )

    def intersection(self, other: "AABB") -> Optional["AABB"]:
        if not self.intersects(other):
            return None
        return AABB(
            max(self.z0, other.z0),
            max(self.y0, other.y0),
            max(self.x0, other.x0),
            min(self.z1, other.z1),
            min(self.y1, other.y1),
            min(self.x1, other.x1),
        )

    def expand(self, halo: int | Sequence[int]) -> "AABB":
        if isinstance(halo, int):
            hz = hy = hx = int(halo)
        else:
            hz, hy, hx = _as_int3(halo)
        return AABB(
            self.z0 - hz,
            self.y0 - hy,
            self.x0 - hx,
            self.z1 + hz,
            self.y1 + hy,
            self.x1 + hx,
        )

    def clip(self, bounds: "AABB") -> Optional["AABB"]:
        return self.intersection(bounds)

    def contains_point(self, z: int, y: int, x: int) -> bool:
        return self.z0 <= z < self.z1 and self.y0 <= y < self.y1 and self.x0 <= x < self.x1

    def to_dict(self) -> dict:
        return {
            "z0": self.z0,
            "y0": self.y0,
            "x0": self.x0,
            "z1": self.z1,
            "y1": self.y1,
            "x1": self.x1,
        }


def overlap_volume(a: AABB, b: AABB) -> int:
    """Voxel count of AABB intersection (0 if disjoint)."""
    inter = a.intersection(b)
    return 0 if inter is None else inter.volume()


def _part1by2(n: int) -> int:
    """Insert two zero bits between each bit of a non-negative 21-bit int."""
    n &= 0x1FFFFF
    n = (n | (n << 32)) & 0x1F00000000FFFF
    n = (n | (n << 16)) & 0x1F0000FF0000FF
    n = (n | (n << 8)) & 0x100F00F00F00F00F
    n = (n | (n << 4)) & 0x10C30C30C30C30C3
    n = (n | (n << 2)) & 0x1249249249249249
    return n


def morton_key(z: int, y: int, x: int) -> int:
    """3D Morton / Z-order key for non-negative ZYX grid indices."""
    if z < 0 or y < 0 or x < 0:
        raise ValueError("morton_key requires non-negative coordinates")
    return _part1by2(z) | (_part1by2(y) << 1) | (_part1by2(x) << 2)


def grid_coords(shape: Sequence[int]) -> np.ndarray:
    """Return (N, 3) array of all ZYX integer coordinates for a volume shape."""
    sz, sy, sx = _as_int3(shape)
    zz, yy, xx = np.meshgrid(
        np.arange(sz, dtype=np.int64),
        np.arange(sy, dtype=np.int64),
        np.arange(sx, dtype=np.int64),
        indexing="ij",
    )
    return np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)


def world_to_tile(
    z: int,
    y: int,
    x: int,
    tile_shape: Sequence[int],
    origin: Sequence[int] = (0, 0, 0),
) -> Coord3:
    oz, oy, ox = _as_int3(origin)
    tz, ty, tx = _as_int3(tile_shape)
    if tz <= 0 or ty <= 0 or tx <= 0:
        raise ValueError("tile_shape must be positive")
    return ((z - oz) // tz, (y - oy) // ty, (x - ox) // tx)


def tile_to_aabb(
    iz: int,
    iy: int,
    ix: int,
    tile_shape: Sequence[int],
    origin: Sequence[int] = (0, 0, 0),
    volume_bounds: Optional[AABB] = None,
) -> AABB:
    oz, oy, ox = _as_int3(origin)
    tz, ty, tx = _as_int3(tile_shape)
    box = AABB(
        oz + iz * tz,
        oy + iy * ty,
        ox + ix * tx,
        oz + (iz + 1) * tz,
        oy + (iy + 1) * ty,
        ox + (ix + 1) * tx,
    )
    if volume_bounds is not None:
        clipped = box.clip(volume_bounds)
        if clipped is None:
            raise ValueError("tile outside volume bounds")
        return clipped
    return box


class TileIndex:
    """ZYX tile index over a volume with optional halo neighbor queries."""

    def __init__(
        self,
        volume_shape: Sequence[int],
        tile_shape: Sequence[int],
        *,
        origin: Sequence[int] = (0, 0, 0),
    ) -> None:
        self.origin = _as_int3(origin)
        self.volume_shape = _as_int3(volume_shape)
        self.tile_shape = _as_int3(tile_shape)
        if any(t <= 0 for t in self.tile_shape):
            raise ValueError("tile_shape must be positive in each axis")
        if any(s <= 0 for s in self.volume_shape):
            raise ValueError("volume_shape must be positive in each axis")
        oz, oy, ox = self.origin
        sz, sy, sx = self.volume_shape
        self.bounds = AABB(oz, oy, ox, oz + sz, oy + sy, ox + sx)
        tz, ty, tx = self.tile_shape
        self.grid_shape = (
            int(np.ceil(sz / tz)),
            int(np.ceil(sy / ty)),
            int(np.ceil(sx / tx)),
        )

    @property
    def n_tiles(self) -> int:
        gz, gy, gx = self.grid_shape
        return gz * gy * gx

    def tile_id(self, iz: int, iy: int, ix: int) -> int:
        gz, gy, gx = self.grid_shape
        if not (0 <= iz < gz and 0 <= iy < gy and 0 <= ix < gx):
            raise IndexError(f"tile index out of range: {(iz, iy, ix)}")
        return (iz * gy + iy) * gx + ix

    def tile_coord(self, tile_id: int) -> Coord3:
        gz, gy, gx = self.grid_shape
        if not (0 <= tile_id < self.n_tiles):
            raise IndexError(f"tile_id out of range: {tile_id}")
        ix = tile_id % gx
        iy = (tile_id // gx) % gy
        iz = tile_id // (gx * gy)
        return (iz, iy, ix)

    def morton_of(self, iz: int, iy: int, ix: int) -> int:
        return morton_key(iz, iy, ix)

    def aabb_of(self, iz: int, iy: int, ix: int) -> AABB:
        return tile_to_aabb(
            iz, iy, ix, self.tile_shape, origin=self.origin, volume_bounds=self.bounds
        )

    def tile_at_world(self, z: int, y: int, x: int) -> Coord3:
        if not self.bounds.contains_point(z, y, x):
            raise IndexError("world coordinate outside volume")
        return world_to_tile(z, y, x, self.tile_shape, origin=self.origin)

    def iter_tiles(self) -> Iterator[Coord3]:
        gz, gy, gx = self.grid_shape
        for iz in range(gz):
            for iy in range(gy):
                for ix in range(gx):
                    yield (iz, iy, ix)

    def neighbors(
        self,
        iz: int,
        iy: int,
        ix: int,
        *,
        halo: int = 1,
        include_self: bool = False,
        only_existing: bool = True,
    ) -> List[Coord3]:
        """Discover neighbor tiles within Chebyshev radius `halo` in tile space."""
        if halo < 0:
            raise ValueError("halo must be >= 0")
        gz, gy, gx = self.grid_shape
        out: List[Coord3] = []
        for dz in range(-halo, halo + 1):
            for dy in range(-halo, halo + 1):
                for dx in range(-halo, halo + 1):
                    if not include_self and dz == 0 and dy == 0 and dx == 0:
                        continue
                    jz, jy, jx = iz + dz, iy + dy, ix + dx
                    if only_existing and not (0 <= jz < gz and 0 <= jy < gy and 0 <= jx < gx):
                        continue
                    out.append((jz, jy, jx))
        return out

    def halo_aabb(self, iz: int, iy: int, ix: int, halo_voxels: int | Sequence[int]) -> AABB:
        core = self.aabb_of(iz, iy, ix)
        expanded = core.expand(halo_voxels)
        clipped = expanded.clip(self.bounds)
        assert clipped is not None
        return clipped

    def overlapping_tiles(self, box: AABB) -> List[Coord3]:
        clipped = box.clip(self.bounds)
        if clipped is None:
            return []
        tz, ty, tx = self.tile_shape
        oz, oy, ox = self.origin
        iz0 = max(0, (clipped.z0 - oz) // tz)
        iy0 = max(0, (clipped.y0 - oy) // ty)
        ix0 = max(0, (clipped.x0 - ox) // tx)
        iz1 = min(self.grid_shape[0] - 1, (clipped.z1 - 1 - oz) // tz)
        iy1 = min(self.grid_shape[1] - 1, (clipped.y1 - 1 - oy) // ty)
        ix1 = min(self.grid_shape[2] - 1, (clipped.x1 - 1 - ox) // tx)
        return [
            (iz, iy, ix)
            for iz in range(iz0, iz1 + 1)
            for iy in range(iy0, iy1 + 1)
            for ix in range(ix0, ix1 + 1)
        ]

    def sorted_by_morton(self) -> List[Coord3]:
        tiles = list(self.iter_tiles())
        tiles.sort(key=lambda t: morton_key(*t))
        return tiles
