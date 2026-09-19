"""Frozen synthetic EM-like regions under receipts/regionbench/."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "receipts" / "regionbench"

# Deterministic frozen set — do not change seeds/shapes without bumping schema.
REGIONS: dict[str, dict[str, Any]] = {
    "rb-empty-032": {"seed": 101, "shape": (32, 32, 32), "kind": "empty", "description": "near-empty tissue"},
    "rb-texture-048": {"seed": 202, "shape": (48, 48, 48), "kind": "texture", "description": "noisy texture field"},
    "rb-blob-064": {"seed": 303, "shape": (64, 64, 64), "kind": "blob", "description": "dense central blob"},
    "rb-seam-040": {"seed": 404, "shape": (40, 40, 40), "kind": "seam", "description": "axial seam discontinuity"},
}


@dataclass
class RegionSpec:
    region_id: str
    seed: int
    shape: tuple[int, int, int]
    kind: str
    description: str
    volume_sha256: str = ""
    npy_relpath: str = ""
    manifest_relpath: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegionRunResult:
    region_id: str
    tool: str
    metrics: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _synthesize(kind: str, shape: tuple[int, int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z, y, x = shape
    if kind == "empty":
        vol = rng.integers(0, 8, size=shape, dtype=np.uint8)
    elif kind == "texture":
        vol = rng.integers(40, 180, size=shape, dtype=np.uint8)
    elif kind == "blob":
        vol = rng.integers(20, 60, size=shape, dtype=np.uint8)
        zz, yy, xx = np.ogrid[:z, :y, :x]
        mask = ((zz - z / 2) ** 2 + (yy - y / 2) ** 2 + (xx - x / 2) ** 2) < (min(shape) / 3.5) ** 2
        vol[mask] = rng.integers(140, 220, size=int(mask.sum()), dtype=np.uint8)
    elif kind == "seam":
        vol = rng.integers(50, 120, size=shape, dtype=np.uint8)
        mid = x // 2
        vol[:, :, mid:] = np.clip(vol[:, :, mid:].astype(np.int16) + 40, 0, 255).astype(np.uint8)
        vol[:, :, mid - 1 : mid + 1] = 10
    else:
        raise ValueError(kind)
    return vol


def ensure_frozen(*, force: bool = False) -> list[RegionSpec]:
    """Materialize frozen npy + manifests; idempotent unless force."""
    ROOT.mkdir(parents=True, exist_ok=True)
    index: list[RegionSpec] = []
    for rid, meta in REGIONS.items():
        rdir = ROOT / rid
        rdir.mkdir(parents=True, exist_ok=True)
        npy_path = rdir / "volume.npy"
        man_path = rdir / "manifest.json"
        if force or not npy_path.exists():
            vol = _synthesize(meta["kind"], tuple(meta["shape"]), int(meta["seed"]))
            np.save(npy_path, vol, allow_pickle=False)
        raw = npy_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        # also hash array content stably
        arr = np.load(npy_path, allow_pickle=False)
        content = hashlib.sha256(arr.tobytes()).hexdigest()
        spec = RegionSpec(
            region_id=rid,
            seed=int(meta["seed"]),
            shape=tuple(meta["shape"]),
            kind=str(meta["kind"]),
            description=str(meta["description"]),
            volume_sha256=content,
            npy_relpath=str(npy_path.relative_to(REPO)).replace("\\", "/"),
            manifest_relpath=str(man_path.relative_to(REPO)).replace("\\", "/"),
        )
        man = {
            "schema": "regionbench/v1",
            "region": spec.as_dict(),
            "file_sha256": digest,
            "frozen": True,
        }
        man_path.write_text(json.dumps(man, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        index.append(spec)
    (ROOT / "index.json").write_text(
        json.dumps({"schema": "regionbench-index/v1", "regions": [s.as_dict() for s in index]}, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    return index


def list_regions() -> list[RegionSpec]:
    if not (ROOT / "index.json").exists():
        return ensure_frozen()
    raw = json.loads((ROOT / "index.json").read_text(encoding="utf-8"))
    out = []
    for r in raw.get("regions", []):
        out.append(RegionSpec(
            region_id=r["region_id"], seed=r["seed"], shape=tuple(r["shape"]),
            kind=r["kind"], description=r["description"],
            volume_sha256=r.get("volume_sha256", ""),
            npy_relpath=r.get("npy_relpath", ""),
            manifest_relpath=r.get("manifest_relpath", ""),
        ))
    return out


def load_volume(region_id: str) -> np.ndarray:
    ensure_frozen()
    path = ROOT / region_id / "volume.npy"
    if not path.exists():
        raise FileNotFoundError(region_id)
    return np.load(path, allow_pickle=False)


class RegionBench:
    def __init__(self) -> None:
        self.regions = ensure_frozen()

    def run(
        self,
        tool: str,
        fn: Callable[[str, np.ndarray], dict[str, Any]],
        *,
        region_ids: list[str] | None = None,
    ) -> list[RegionRunResult]:
        ids = region_ids or [s.region_id for s in self.regions]
        results: list[RegionRunResult] = []
        for rid in ids:
            vol = load_volume(rid)
            try:
                metrics = fn(rid, vol)
                results.append(RegionRunResult(rid, tool, metrics, True))
            except Exception as exc:
                results.append(RegionRunResult(rid, tool, {}, False, str(exc)))
        out = ROOT / f"results_{tool}.json"
        out.write_text(
            json.dumps([r.as_dict() for r in results], indent=2, sort_keys=True) + "\\n",
            encoding="utf-8",
        )
        return results


def run_region(region_id: str, tool: str, fn: Callable[[str, np.ndarray], dict[str, Any]]) -> RegionRunResult:
    return RegionBench().run(tool, fn, region_ids=[region_id])[0]
