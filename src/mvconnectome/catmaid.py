"""Read-only ingestion of documented public CATMAID/DVID image tiles."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image

from .io import sha256_file, write_json_atomic


def load_region(path: Path) -> dict:
    """Load the documented region request rather than infer tile geometry."""
    # JSON is a valid YAML subset; this keeps the manifest dependency-free.
    return json.loads(path.read_text(encoding="utf-8"))


def tile_url(region: dict, x: int, y: int, z: int) -> str:
    """Construct one CATMAID tile request from an approved region description."""
    width, height = region["tile_size_xy"]
    return f"{region['source_tile_url']}xy/0/{x // width}_{y // height}_{z}"


def fetch_region(region_path: Path, cache_root: Path) -> Path:
    """Fetch declared public tiles into cache, retaining the request provenance."""
    region = load_region(region_path)
    bounds = region["source_bounds_xyz"]
    x0, x1 = bounds["x"]; y0, y1 = bounds["y"]; z0, z1 = bounds["z"]
    tile_width, tile_height = region["tile_size_xy"]
    if x0 % tile_width or x1 % tile_width or y0 % tile_height or y1 % tile_height:
        raise ValueError("Phase 2 ingester currently requires tile-aligned XY bounds")
    destination = cache_root / region["volume_id"] / region["id"]
    tiles = destination / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "raw_zyx.npy"
    volume = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.uint8, shape=(z1-z0, y1-y0, x1-x0))
    for z in range(z0, z1):
        for y in range(y0, y1, tile_height):
            for x in range(x0, x1, tile_width):
                tile_path = tiles / f"xy_x{x}_y{y}_z{z}.jpg"
                if not tile_path.exists():
                    with urlopen(tile_url(region, x, y, z), timeout=60) as response:
                        payload = response.read()
                    if not payload.startswith(b"\xff\xd8"):
                        raise ValueError(f"Unexpected non-JPEG response at {x},{y},{z}")
                    tile_path.write_bytes(payload)
                with Image.open(tile_path) as image:
                    array = np.asarray(image.convert("L"), dtype=np.uint8)
                if array.shape != (tile_height, tile_width):
                    raise ValueError(f"Unexpected tile shape {array.shape} at {x},{y},{z}")
                volume[z-z0, y-y0:y-y0+tile_height, x-x0:x-x0+tile_width] = array
    volume.flush()
    record = {
        "schema_version": "1.0", "kind": "remote-tile-volume-ingestion", "region_id": region["id"],
        "volume_id": region["volume_id"], "dataset_id": region["dataset_id"], "source_bounds_xyz": bounds,
        "shape_zyx": [z1-z0, y1-y0, x1-x0], "raw_array": str(raw_path.resolve()), "raw_sha256": sha256_file(raw_path),
        "tile_count": ((x1-x0)//tile_width)*((y1-y0)//tile_height)*(z1-z0), "source_manifest": str(region_path.resolve()),
        "raw_data_immutable": True, "local_cache_only": True,
    }
    write_json_atomic(destination / "ingestion.json", record)
    return destination / "ingestion.json"


def fetch_dvid_raw_region(region_path: Path, cache_root: Path) -> Path:
    """Fetch one bounded DVID raw volume without manufacturing metadata or labels."""
    """Fetch one bounded raw DVID subvolume; source remains external/local-cache-only."""
    region = load_region(region_path)
    bounds = region["source_bounds_xyz"]
    x0, x1 = bounds["x"]; y0, y1 = bounds["y"]; z0, z1 = bounds["z"]
    size = (x1-x0, y1-y0, z1-z0)
    url = f"{region['source_raw_url']}{size[0]}_{size[1]}_{size[2]}/{x0}_{y0}_{z0}"
    destination = cache_root / region["volume_id"] / region["id"]
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "raw_zyx.npy"
    with urlopen(url, timeout=180) as response:
        payload = response.read()
    expected_bytes = size[0] * size[1] * size[2]
    if len(payload) != expected_bytes:
        raise ValueError(f"DVID raw response size {len(payload)} != expected {expected_bytes}")
    np.save(raw_path, np.frombuffer(payload, dtype=np.uint8).reshape((size[2], size[1], size[0])))
    record = {
        "schema_version": "1.0", "kind": "remote-dvid-raw-volume-ingestion", "region_id": region["id"], "volume_id": region["volume_id"],
        "dataset_id": region["dataset_id"], "source_bounds_xyz": bounds, "shape_zyx": [size[2], size[1], size[0]],
        "raw_array": str(raw_path.resolve()), "raw_sha256": sha256_file(raw_path), "source_url": url,
        "source_node": region["source_node"], "source_instance": region["source_instance"], "raw_data_immutable": True, "local_cache_only": True,
    }
    write_json_atomic(destination / "ingestion.json", record)
    return destination / "ingestion.json"
