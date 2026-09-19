"""Adapters for auditable bounded SegNeuron technical inference.

This is deliberately limited to preparing a real DVID smoke-input receipt.
It does not promote affinities or an instance result to an MV-SEG record.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .io import sha256_file, write_json_atomic


def export_dvid_smoke_input(ingestion: Path, output: Path, receipt: Path, start_zyx: tuple[int, int, int] = (0, 0, 0), shape_zyx: tuple[int, int, int] = (20, 128, 128)) -> dict:
    import numpy as np

    metadata = json.loads(ingestion.read_text(encoding="utf-8"))
    raw_path = ingestion.parent / "raw_zyx.npy"
    raw = np.load(raw_path, allow_pickle=False, mmap_mode="r")
    z0, y0, x0 = start_zyx
    dz, dy, dx = shape_zyx
    if raw.ndim != 3 or raw.dtype != np.uint8:
        raise ValueError("DVID smoke source must be a uint8 ZYX raw volume")
    if min(z0, y0, x0) < 0 or z0 + dz > raw.shape[0] or y0 + dy > raw.shape[1] or x0 + dx > raw.shape[2]:
        raise ValueError("Smoke bounds are outside the immutable cached DVID crop")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"Refusing to overwrite smoke input: {output}")
    value = np.asarray(raw[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx])
    np.save(output, value, allow_pickle=False)
    source_bounds = metadata["source_bounds_xyz"]
    x = source_bounds["x"][0] + x0
    y = source_bounds["y"][0] + y0
    z = source_bounds["z"][0] + z0
    record = {
        "id": "MV-INPUT-DVID-SMOKE-0001", "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "dataset_id": metadata["dataset_id"], "volume_id": metadata["volume_id"], "coordinate_frame": "MV-FRAME-DVID-WASP5-001",
        "source_ingestion": str(ingestion), "source_raw_sha256": metadata["raw_sha256"],
        "source_bounds_xyz": {"x": [x, x + dx], "y": [y, y + dy], "z": [z, z + dz]},
        "array_bounds_zyx": {"z": [z0, z0 + dz], "y": [y0, y0 + dy], "x": [x0, x0 + dx]},
        "shape_zyx": list(value.shape), "dtype": str(value.dtype), "sha256": sha256_file(output),
        "purpose": "TECHNICAL_EXECUTION_SMOKE_ONLY_NOT_A_VALIDATION_OR_PRODUCTION_SEGMENTATION_INPUT",
        "raw_immutable": True,
    }
    write_json_atomic(receipt, record)
    return record
