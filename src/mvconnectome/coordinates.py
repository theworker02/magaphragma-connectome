"""Coordinate-frame declarations and explicit conversion utilities.

The project never guesses whether a tuple is XYZ, ZYX, physical nanometres,
or voxels. This prevents a quiet axis swap becoming a false alignment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoordinateTransform:
    """Explicit source/internal/viewer transform; avoids implicit XYZ/ZYX conversions."""
    origin_nm_xyz: tuple[float, float, float]
    resolution_nm_xyz: tuple[float, float, float]

    def source_to_internal(self, point_xyz: tuple[int, int, int]) -> tuple[int, int, int]:
        return point_xyz

    def internal_to_source(self, point_xyz: tuple[int, int, int]) -> tuple[int, int, int]:
        return point_xyz

    def internal_to_physical(self, point_xyz: tuple[int, int, int]) -> tuple[float, float, float]:
        return tuple(origin + coordinate * resolution for origin, coordinate, resolution in zip(self.origin_nm_xyz, point_xyz, self.resolution_nm_xyz, strict=True))

    def physical_to_internal(self, point_nm_xyz: tuple[float, float, float]) -> tuple[int, int, int]:
        return tuple(round((point - origin) / resolution) for point, origin, resolution in zip(point_nm_xyz, self.origin_nm_xyz, self.resolution_nm_xyz, strict=True))

    def viewer_to_internal(self, point_xyz: tuple[int, int, int]) -> tuple[int, int, int]:
        return point_xyz

    def internal_to_viewer(self, point_xyz: tuple[int, int, int]) -> tuple[int, int, int]:
        return point_xyz
