"""PriorityMap ? per-tile scout results with JSON IO and AxonForge hints."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TilePriority:
    tile_key: str
    region_class: str
    tissue_probability: float
    information_density: float
    boundary_complexity: float
    artifact_score: float
    uncertainty: float
    prior_coverage: float
    recommend_priority: float
    box: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PriorityMap:
    """Complete tile map ? every tile accounted; never discarded."""

    volume_shape: tuple[int, int, int]
    tile_zyx: tuple[int, int, int]
    tiles: list[TilePriority] = field(default_factory=list)
    tool_version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "volume_shape": list(self.volume_shape),
            "tile_zyx": list(self.tile_zyx),
            "tool_version": self.tool_version,
            "n_tiles": len(self.tiles),
            "tiles": [t.to_dict() for t in self.tiles],
            "class_counts": self.class_counts(),
        }

    def save(self, path: Path | None = None, *, receipts_root: Path | None = None) -> Path:
        if path is None:
            root = receipts_root or Path("receipts") / "voxscout"
            root.mkdir(parents=True, exist_ok=True)
            path = root / "priority_map.json"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + chr(10), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "PriorityMap":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tiles = [TilePriority(**{k: v for k, v in t.items() if k in TilePriority.__dataclass_fields__}) for t in data["tiles"]]
        return cls(
            volume_shape=tuple(data["volume_shape"]),
            tile_zyx=tuple(data["tile_zyx"]),
            tiles=tiles,
            tool_version=str(data.get("tool_version", "0.1.0")),
        )

    def as_axonforge_hints(self) -> dict[str, dict[str, Any]]:
        ok_classes = {"EMPTY", "LOW_INFORMATION"}
        out: dict[str, dict[str, Any]] = {}
        for t in self.tiles:
            out[t.tile_key] = {
                "priority": float(t.recommend_priority),
                "class": t.region_class,
                "optimize_ok": t.region_class in ok_classes,
            }
        return out

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.tiles:
            counts[t.region_class] = counts.get(t.region_class, 0) + 1
        return counts
