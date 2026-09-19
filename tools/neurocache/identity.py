"""Scientific identity hashing for cache keys."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

EXCLUDED_KEYS = frozenset(
    {
        "logging_verbosity",
        "log_level",
        "verbose",
        "debug",
        "threads",
        "num_workers",
        "device",
        "gpu_id",
        "wall_clock",
        "hostname",
        "pid",
    }
)


@dataclass(frozen=True)
class ScientificIdentity:
    """Canonical scientific identity for a computation."""

    source_volume_hash: str
    coordinates: tuple[float, ...] | list[float]
    resolution: tuple[float, ...] | list[float]
    preprocessing: dict[str, Any] | str
    model: str
    checkpoint: str
    precision: str
    tile_dimensions: tuple[int, ...] | list[int]
    halo: tuple[int, ...] | list[int] | int
    scientific_config: dict[str, Any] = field(default_factory=dict)
    algorithm_version: str = "1.0"

    def canonical_dict(self) -> dict[str, Any]:
        def scrub(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    str(k): scrub(v)
                    for k, v in sorted(obj.items())
                    if str(k) not in EXCLUDED_KEYS
                }
            if isinstance(obj, (list, tuple)):
                return [scrub(v) for v in obj]
            return obj

        raw = {
            "algorithm_version": self.algorithm_version,
            "checkpoint": self.checkpoint,
            "coordinates": list(self.coordinates),
            "halo": self.halo if isinstance(self.halo, int) else list(self.halo),
            "model": self.model,
            "precision": self.precision,
            "preprocessing": scrub(self.preprocessing),
            "resolution": list(self.resolution),
            "scientific_config": scrub(self.scientific_config),
            "source_volume_hash": self.source_volume_hash,
            "tile_dimensions": list(self.tile_dimensions),
        }
        return scrub(raw)

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def identity_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["coordinates"] = list(self.coordinates)
        d["resolution"] = list(self.resolution)
        d["tile_dimensions"] = list(self.tile_dimensions)
        if not isinstance(self.halo, int):
            d["halo"] = list(self.halo)
        d["identity_hash"] = self.identity_hash()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScientificIdentity":
        return cls(
            source_volume_hash=str(data["source_volume_hash"]),
            coordinates=list(data["coordinates"]),
            resolution=list(data["resolution"]),
            preprocessing=data.get("preprocessing", {}),
            model=str(data["model"]),
            checkpoint=str(data["checkpoint"]),
            precision=str(data.get("precision", "fp32")),
            tile_dimensions=list(data["tile_dimensions"]),
            halo=data.get("halo", 0),
            scientific_config=dict(data.get("scientific_config") or {}),
            algorithm_version=str(data.get("algorithm_version", "1.0")),
        )
