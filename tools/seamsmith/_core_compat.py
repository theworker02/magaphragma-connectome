"""Import tools._core when present; otherwise local fallbacks matching its API."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class RegionClass(str, Enum):
    EMPTY = "EMPTY"
    LOW_INFORMATION = "LOW_INFORMATION"
    NORMAL = "NORMAL"
    HIGH_COMPLEXITY = "HIGH_COMPLEXITY"
    UNCERTAIN = "UNCERTAIN"
    ARTIFACT = "ARTIFACT"
    REPROCESS = "REPROCESS"


@dataclass(frozen=True)
class AABB:
    z0: int
    y0: int
    x0: int
    z1: int
    y1: int
    x1: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.z1 - self.z0, self.y1 - self.y0, self.x1 - self.x0)

    def as_dict(self) -> dict:
        return {"z0": self.z0, "y0": self.y0, "x0": self.x0, "z1": self.z1, "y1": self.y1, "x1": self.x1}


@dataclass
class TileIndex:
    shape_zyx: tuple[int, int, int]
    tile_zyx: tuple[int, int, int] = (32, 64, 64)

    def iter_tiles(self) -> Iterator[tuple[str, AABB]]:
        tz, ty, tx = self.tile_zyx
        Z, Y, X = self.shape_zyx
        for z in range(0, Z, tz):
            for y in range(0, Y, ty):
                for x in range(0, X, tx):
                    box = AABB(z, y, x, min(z + tz, Z), min(y + ty, Y), min(x + tx, X))
                    iz, iy, ix = z // tz, y // ty, x // tx
                    key = f"T-Z{iz:03d}-Y{iy:03d}-X{ix:03d}"
                    yield key, box

    def crop(self, volume, box: AABB):
        return volume[box.z0 : box.z1, box.y0 : box.y1, box.x0 : box.x1]


def write_tool_receipt(tool: str, payload: dict[str, Any], *, repo: Path | None = None) -> Path:
    base = Path(repo) if repo is not None else Path(".")
    root = base / "receipts" / "tools" / tool
    root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    body = {"tool": tool, "ts": ts, **payload}
    raw = json.dumps(body, indent=2, sort_keys=True) + chr(10)
    stamped = root / f"{ts}.json"
    stamped.write_text(raw, encoding="utf-8")
    (root / "latest.json").write_text(raw, encoding="utf-8")
    return stamped


@dataclass
class ArtifactRef:
    artifact_id: str
    kind: str
    sha256: str
    path: str
    meta: dict


@dataclass
class ArtifactStore:
    root: Path | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root) if self.root else Path("receipts") / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)

    def put_json(self, kind: str, payload: dict[str, Any], *, meta: dict | None = None) -> ArtifactRef:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()
        aid = f"{kind}-{sha[:16]}"
        dest = self.root / kind / sha
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "payload.json").write_bytes(raw + bytes([10]))
        m = {"kind": kind, **(meta or {})}
        (dest / "manifest.json").write_text(json.dumps(m, indent=2) + chr(10), encoding="utf-8")
        return ArtifactRef(aid, kind, sha, str(dest / "payload.json"), m)

    def get_json(self, sha256: str, kind: str) -> dict:
        return json.loads((self.root / kind / sha256 / "payload.json").read_text(encoding="utf-8"))


class ProvenanceGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.fwd: dict[str, list[tuple[str, str]]] = {}

    def add_node(self, node_id: str, kind: str, **meta: Any) -> dict[str, Any]:
        self.nodes[node_id] = {"kind": kind, "meta": meta}
        return self.nodes[node_id]

    def link(self, parent: str, child: str, relation: str = "produces") -> None:
        self.fwd.setdefault(parent, []).append((child, relation))

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [
                {"parent": p, "child": c, "relation": r}
                for p, lst in self.fwd.items()
                for c, r in lst
            ],
        }


def load_core():
    try:
        from _core import (  # type: ignore
            ArtifactStore as AS,
            ProvenanceGraph as PG,
            RegionClass as RC,
            TileIndex as TI,
            write_tool_receipt as WTR,
        )
        return RC, TI, WTR, AS, PG
    except Exception:
        return RegionClass, TileIndex, write_tool_receipt, ArtifactStore, ProvenanceGraph
