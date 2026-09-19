"""Spatial work graph — immutable tile map, mutable execution schedule."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class TileId:
    x: int
    y: int
    z: int
    level: int = 0

    @property
    def key(self) -> str:
        return f"AF-VOLUME-X{self.x:03d}-Y{self.y:03d}-Z{self.z:03d}-L{self.level}"


@dataclass
class TileTask:
    tile: TileId
    status: str = "PENDING"  # PENDING|RUNNING|VALIDATED|REJECTED|CACHED
    priority: float = 0.0
    reason: str = ""
    voxels: int = 0


@dataclass
class SpatialWorkGraph:
    shape_zyx: tuple[int, int, int]
    tile_zyx: tuple[int, int, int] = (32, 64, 64)
    tasks: dict[str, TileTask] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def build(
        cls,
        shape_zyx: tuple[int, int, int],
        tile_zyx: tuple[int, int, int] = (32, 64, 64),
        path: Path | None = None,
        load_existing: bool = True,
    ) -> "SpatialWorkGraph":
        g = cls(shape_zyx=shape_zyx, tile_zyx=tile_zyx, path=path)
        if load_existing and path and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            g.tasks = {
                k: TileTask(
                    tile=TileId(**v["tile"]),
                    status=v["status"],
                    priority=v.get("priority", 0.0),
                    reason=v.get("reason", ""),
                    voxels=v.get("voxels", 0),
                )
                for k, v in raw.get("tasks", {}).items()
            }
            return g
        tz, ty, tx = tile_zyx
        z, y, x = shape_zyx
        for zi in range(0, z, tz):
            for yi in range(0, y, ty):
                for xi in range(0, x, tx):
                    tid = TileId(x=xi // tx, y=yi // ty, z=zi // tz)
                    vz = min(tz, z - zi) * min(ty, y - yi) * min(tx, x - xi)
                    g.tasks[tid.key] = TileTask(tile=tid, voxels=int(vz))
        return g

    def iter_pending(self) -> Iterator[TileTask]:
        for t in self.tasks.values():
            if t.status == "PENDING":
                yield t

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "shape_zyx": list(self.shape_zyx),
            "tile_zyx": list(self.tile_zyx),
            "tasks": {
                k: {
                    "tile": asdict(v.tile),
                    "status": v.status,
                    "priority": v.priority,
                    "reason": v.reason,
                    "voxels": v.voxels,
                }
                for k, v in self.tasks.items()
            },
        }
        self.path.write_text(json.dumps(payload, indent=2) + chr(10), encoding="utf-8")
