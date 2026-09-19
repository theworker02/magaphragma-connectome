"""Affinity cost model: dollars per completed production chunk."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

TILES_PER_CHUNK = 3468
BASELINE_TILES_PER_SEC = 18.9  # RX 7800 XT packed-eager reference


@dataclass
class CostModel:
    tiles_per_chunk: int = TILES_PER_CHUNK
    validated_tiles_per_second: float = BASELINE_TILES_PER_SEC
    gpu_hourly_usd: float = 0.0
    storage_hourly_usd: float = 0.0
    bandwidth_per_chunk_usd: float = 0.0
    remaining_chunks: int = 0

    @property
    def seconds_per_chunk(self) -> float:
        if self.validated_tiles_per_second <= 0:
            return float("inf")
        return self.tiles_per_chunk / self.validated_tiles_per_second

    @property
    def chunks_per_hour(self) -> float:
        spc = self.seconds_per_chunk
        if spc <= 0 or spc == float("inf"):
            return 0.0
        return 3600.0 / spc

    @property
    def compute_cost_per_chunk(self) -> float:
        cph = self.chunks_per_hour
        if cph <= 0:
            return float("inf")
        return (self.gpu_hourly_usd + self.storage_hourly_usd) / cph

    @property
    def effective_cost_per_chunk(self) -> float:
        return self.compute_cost_per_chunk + self.bandwidth_per_chunk_usd

    @property
    def projected_remaining_cost_usd(self) -> float:
        return self.remaining_chunks * self.effective_cost_per_chunk

    @property
    def projected_remaining_gpu_hours(self) -> float:
        if self.validated_tiles_per_second <= 0:
            return float("inf")
        return (self.remaining_chunks * self.tiles_per_chunk) / self.validated_tiles_per_second / 3600.0

    @property
    def cost_per_1000_chunks(self) -> float:
        return self.effective_cost_per_chunk * 1000.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            {
                "seconds_per_chunk": self.seconds_per_chunk,
                "chunks_per_hour": self.chunks_per_hour,
                "compute_cost_per_chunk": self.compute_cost_per_chunk,
                "effective_cost_per_chunk": self.effective_cost_per_chunk,
                "projected_remaining_cost_usd": self.projected_remaining_cost_usd,
                "projected_remaining_gpu_hours": self.projected_remaining_gpu_hours,
                "cost_per_1000_chunks": self.cost_per_1000_chunks,
            }
        )
        return d


def estimate_chunk_economics(
    *,
    tiles_per_second: float,
    gpu_hourly_usd: float,
    remaining_chunks: int,
    storage_hourly_usd: float = 0.0,
    bandwidth_per_chunk_usd: float = 0.0,
    tiles_per_chunk: int = TILES_PER_CHUNK,
) -> CostModel:
    return CostModel(
        tiles_per_chunk=tiles_per_chunk,
        validated_tiles_per_second=tiles_per_second,
        gpu_hourly_usd=gpu_hourly_usd,
        storage_hourly_usd=storage_hourly_usd,
        bandwidth_per_chunk_usd=bandwidth_per_chunk_usd,
        remaining_chunks=remaining_chunks,
    )


def fleet_size_for_target_hours(
    *,
    remaining_chunks: int,
    tiles_per_second_per_gpu: float,
    target_hours: float,
    max_gpus: int,
    tiles_per_chunk: int = TILES_PER_CHUNK,
) -> int:
    """Minimum GPU count to finish within target_hours at measured per-GPU TPS."""
    if tiles_per_second_per_gpu <= 0 or target_hours <= 0:
        return 0
    tiles = remaining_chunks * tiles_per_chunk
    need = tiles / tiles_per_second_per_gpu / 3600.0 / target_hours
    n = max(1, int(need + 0.999))
    return min(n, max_gpus)


def max_gpus_under_budget(
    *,
    budget_usd: float,
    hours: float,
    dph_per_gpu: float,
    ancillary_usd: float = 0.0,
) -> int:
    """How many GPUs can run for `hours` without exceeding budget."""
    if dph_per_gpu <= 0 or hours <= 0:
        return 0
    usable = max(0.0, budget_usd - ancillary_usd)
    return int(usable // (dph_per_gpu * hours))
