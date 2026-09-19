"""Evidence-preserving planning for a large DVID reconstruction domain.

This module deliberately separates *reported source extent* from imagery that
has actually been cached.  Creating a work item never asserts that its raw
voxels, a segmentation, or a biological structure have been acquired.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .io import sha256_file, write_json_atomic


PARENT_INFO_URL = (
    "https://waspem-dvid2.flatironinstitute.org/api/node/"
    "aa49d16e88424cdbae89f4b8ced2a49b/five_yuri_4contrast/info"
)
RAW_URL_PREFIX = (
    "https://waspem-dvid2.flatironinstitute.org/api/node/"
    "aa49d16e88424cdbae89f4b8ced2a49b/five_yuri_4contrast/raw/0_1_2"
)


def _now() -> str:
    """Return a UTC timestamp for domain-planning provenance."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    """Atomically retain a fetched source response before parsing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _get(url: str) -> bytes:
    """Retrieve one explicit source endpoint with no implicit fallback source."""
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _bounds_from_info(info: dict[str, Any]) -> dict[str, list[int]]:
    """Convert DVID inclusive points into the project's explicit half-open bounds."""
    extended = info.get("Extended")
    if not isinstance(extended, dict):
        raise ValueError("DVID info lacks Extended metadata")
    minimum = extended.get("MinPoint")
    maximum = extended.get("MaxPoint")
    if not (isinstance(minimum, list) and isinstance(maximum, list) and len(minimum) == len(maximum) == 3):
        raise ValueError("DVID info lacks three-dimensional MinPoint/MaxPoint")
    return {axis: [int(minimum[index]), int(maximum[index]) + 1] for index, axis in enumerate(("x", "y", "z"))}


def _read_bounds(core: dict[str, list[int]], full: dict[str, list[int]], halo: tuple[int, int, int]) -> dict[str, list[int]]:
    """Describe one halo-expanded read region around a resumable core work item."""
    return {
        axis: [max(full[axis][0], core[axis][0] - halo[index]), min(full[axis][1], core[axis][1] + halo[index])]
        for index, axis in enumerate(("x", "y", "z"))
    }


def _chunk_index(bounds: dict[str, list[int]], core_shape: tuple[int, int, int], halo: tuple[int, int, int]) -> list[dict[str, Any]]:
    """Create the durable global queue; each item is only a planned read."""
    counts = [
        (bounds[axis][1] - bounds[axis][0] + core_shape[index] - 1) // core_shape[index]
        for index, axis in enumerate(("x", "y", "z"))
    ]
    chunks: list[dict[str, Any]] = []
    serial = 10_000_001
    for zi in range(counts[2]):
        for yi in range(counts[1]):
            for xi in range(counts[0]):
                indices = (xi, yi, zi)
                core = {
                    axis: [
                        bounds[axis][0] + indices[index] * core_shape[index],
                        min(bounds[axis][1], bounds[axis][0] + (indices[index] + 1) * core_shape[index]),
                    ]
                    for index, axis in enumerate(("x", "y", "z"))
                }
                neighbors = []
                for axis_index, axis in enumerate(("x", "y", "z")):
                    for delta in (-1, 1):
                        adjacent = list(indices)
                        adjacent[axis_index] += delta
                        if 0 <= adjacent[axis_index] < counts[axis_index]:
                            neighbor_serial = 10_000_001 + adjacent[2] * counts[0] * counts[1] + adjacent[1] * counts[0] + adjacent[0]
                            neighbors.append(f"MV-CHUNK-{neighbor_serial:08d}")
                chunks.append({
                    "id": f"MV-CHUNK-{serial:08d}",
                    "domain_id": "MV-DOMAIN-PRODUCTION-001",
                    "specimen_id": "SPECIMEN_UNKNOWN",
                    "volume_id": "MV-FIBSEM-WASP5-YURI-4C",
                    "grid_index_xyz": [xi, yi, zi],
                    "core_bounds_xyz": core,
                    "read_bounds_xyz": _read_bounds(core, bounds, halo),
                    "overlap_margin_voxels_xyz": list(halo),
                    "neighbor_chunk_ids": sorted(neighbors),
                    "source_read_state": "NOT_REQUESTED",
                    "processing_state": "NOT_STARTED",
                    "segmentation_state": "NOT_STARTED",
                    "skeleton_state": "NOT_STARTED",
                    "synapse_state": "NOT_STARTED",
                    "review_state": "UNREVIEWED",
                    "retry_count": 0,
                })
                serial += 1
    return chunks


def inspect_and_plan(output: Path, cache: Path, parent_info_url: str = PARENT_INFO_URL) -> dict[str, Any]:
    """Fetch authoritative metadata and create a non-processing production queue.

    A 64³ final-boundary canary is retained as an immutable cache receipt.  It
    establishes that the reported maximum edge can be read, but it does not
    assert that the entire multi-terabyte domain was downloaded or processed.
    """
    raw_info = _get(parent_info_url)
    info = json.loads(raw_info.decode("utf-8"))
    if not isinstance(info, dict):
        raise ValueError("DVID parent metadata is not an object")
    bounds = _bounds_from_info(info)
    extended = info["Extended"]
    voxel_size = [float(value) for value in extended["VoxelSize"]]
    if voxel_size != [8.0, 8.0, 8.0] or extended.get("VoxelUnits") != ["nanometers"] * 3:
        raise ValueError("Unexpected DVID voxel geometry; refusing to plan with assumed coordinates")
    block_size = [int(value) for value in extended["BlockSize"]]
    if block_size != [64, 64, 64]:
        raise ValueError("Unexpected DVID block shape; update planning rationale explicitly")

    info_path = cache / "MV-FIBSEM-WASP5-YURI-4C" / "parent-info.json"
    _write_bytes_atomic(info_path, raw_info)
    canary_start = [bounds[axis][1] - block_size[index] for index, axis in enumerate(("x", "y", "z"))]
    canary_url = f"{RAW_URL_PREFIX}/64_64_64/{canary_start[0]}_{canary_start[1]}_{canary_start[2]}"
    canary = _get(canary_url)
    expected_canary_bytes = block_size[0] * block_size[1] * block_size[2]
    if len(canary) != expected_canary_bytes:
        raise ValueError(f"DVID boundary canary was {len(canary)} bytes, expected {expected_canary_bytes}")
    canary_path = cache / "MV-FIBSEM-WASP5-YURI-4C" / "production-boundary-canary-zyx.uint8"
    _write_bytes_atomic(canary_path, canary)

    dimensions = [bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z")]
    voxel_count = dimensions[0] * dimensions[1] * dimensions[2]
    core_shape = (1024, 1024, 128)
    halo = (64, 64, 16)
    chunks = _chunk_index(bounds, core_shape, halo)
    domain = {
        "id": "MV-DOMAIN-PRODUCTION-001",
        "kind": "DVID_PARENT_PRODUCTION_DOMAIN",
        "dataset_id": "MV-SRC-FLATIRON-DVID-WASP5",
        "volume_id": "MV-FIBSEM-WASP5-YURI-4C",
        "specimen_id": "SPECIMEN_UNKNOWN",
        "coordinate_frame": "MV-FRAME-DVID-WASP5-001",
        "axis_order": "XYZ",
        "array_order": "ZYX",
        "voxel_indexing": "zero_based_voxel_centers",
        "source_bounds_xyz": bounds,
        "dimensions_voxels_xyz": dimensions,
        "voxel_size_nm_xyz": voxel_size,
        "physical_dimensions_nm_xyz": [dimensions[index] * voxel_size[index] for index in range(3)],
        "estimated_raw_uint8_bytes": voxel_count,
        "source_access": {
            "status": "PUBLIC_PARENT_METADATA_AND_FAR_BOUNDARY_RAW_READ_VERIFIED",
            "parent_metadata_url": parent_info_url,
            "parent_metadata_sha256": sha256_file(info_path),
            "far_boundary_raw_read_url": canary_url,
            "far_boundary_raw_sha256": sha256_file(canary_path),
            "far_boundary_raw_bytes": len(canary),
            "whole_domain_downloaded": False,
            "whole_domain_processed": False,
            "rights": "UNVERIFIED_FOR_REDISTRIBUTION_AND_DERIVATIVE_RELEASE",
            "eligibility": "LOCAL_PROCESSING_ONLY_PENDING_SOURCE_TERMS_REVIEW",
        },
        "processing_policy": {
            "raw_data_immutable": True,
            "no_catmaid_identity_or_seed_transfer": True,
            "catmaid_dvid_transform_status": "INSUFFICIENT_EVIDENCE",
            "production_segmenter_status": "NONE_SELECTED",
            "synapse_backend_status": "UNKNOWN_OR_UNAVAILABLE",
        },
        "chunking": {
            "core_shape_voxels_xyz": list(core_shape),
            "overlap_margin_voxels_xyz": list(halo),
            "source_block_shape_voxels_xyz": block_size,
            "rationale": "1024x1024x128 cores are multiples of DVID 64-cube blocks; 64x64x16 read halos are provisional planning margins and must be revalidated against the selected model receptive field.",
            "chunk_count": len(chunks),
        },
        "created_at": _now(),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "domain.json", domain)
    write_json_atomic(output / "chunks.json", {"chunks": chunks})
    write_json_atomic(output / "source_receipt.json", {
        "parent_info_cache": str(info_path), "parent_info_sha256": sha256_file(info_path),
        "boundary_canary_cache": str(canary_path), "boundary_canary_sha256": sha256_file(canary_path),
        "acquired_at": _now(),
    })
    return {"domain": domain["id"], "dimensions_voxels_xyz": dimensions, "chunk_count": len(chunks), "raw_bytes": voxel_count, "source_access": domain["source_access"]["status"]}
