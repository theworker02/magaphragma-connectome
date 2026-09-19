"""Completeness ledger — every voxel accounted for."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompletenessLedger:
    total_voxels: int = 0
    validated_voxels: int = 0
    rejected_voxels: int = 0
    cached_voxels: int = 0
    uncertain_regions: int = 0

    def record(self, voxels: int, status: str) -> None:
        if status == "VALIDATED":
            self.validated_voxels += voxels
        elif status == "REJECTED":
            self.rejected_voxels += voxels
            self.uncertain_regions += 1
        elif status == "CACHED":
            self.cached_voxels += voxels
            self.validated_voxels += voxels

    @property
    def validated_coverage(self) -> float:
        if self.total_voxels <= 0:
            return 0.0
        return self.validated_voxels / self.total_voxels

    @property
    def unaccounted_volume(self) -> int:
        accounted = self.validated_voxels + self.rejected_voxels
        # cached counted inside validated
        return max(0, self.total_voxels - self.validated_voxels - self.rejected_voxels)

    def as_dict(self) -> dict:
        return {
            "total_voxels": self.total_voxels,
            "validated_voxels": self.validated_voxels,
            "rejected_voxels": self.rejected_voxels,
            "cached_voxels": self.cached_voxels,
            "uncertain_regions": self.uncertain_regions,
            "validated_coverage": self.validated_coverage,
            "unaccounted_volume": self.unaccounted_volume,
        }
