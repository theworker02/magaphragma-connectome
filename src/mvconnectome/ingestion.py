"""Local, checksumed volume registration; no silent coordinate inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, write_json_atomic
from .models import EvidenceStatus, VolumeMetadata, as_json

SUPPORTED_SUFFIXES = {".tif", ".tiff", ".ome.tif", ".ome.tiff", ".zarr", ".n5", ".h5", ".hdf5"}


def _suffix(path: Path) -> str:
    """Preserve compound microscopy suffixes when recording artifact format."""
    return ".".join(path.suffixes[-2:]).lower() if len(path.suffixes) > 1 else path.suffix.lower()


def metadata_from_json(path: Path) -> VolumeMetadata:
    """Parse an authority-supplied sidecar through the VolumeMetadata invariants."""
    payload = read_json(path)
    return VolumeMetadata(
        dataset_id=payload["dataset_id"], specimen_id=payload["specimen_id"], format=payload["format"],
        voxel_size_nm=tuple(payload["voxel_size_nm"]), shape_zyx=tuple(payload["shape_zyx"]),
        coordinate_frame=payload["coordinate_frame"], origin_nm_xyz=tuple(payload["origin_nm_xyz"]),
        source_url=payload["source_url"], source_sha256=payload["source_sha256"],
        evidence_status=EvidenceStatus(payload["evidence_status"]), notes=payload.get("notes"),
    )


def ingest_volume(volume_path: Path, metadata_path: Path, output_root: Path) -> Path:
    """Write immutable registration evidence for a local acquired artifact.

    This function intentionally registers rather than transforms imagery; rechunking and conversion
    must be a separately recorded experiment.
    """
    if not volume_path.exists():
        raise FileNotFoundError(volume_path)
    if _suffix(volume_path) not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported volume format for {volume_path.name}; supported: {sorted(SUPPORTED_SUFFIXES)}")
    if volume_path.is_dir():
        raise ValueError("Directory-backed formats require a deterministic archive manifest before ingestion")
    metadata = metadata_from_json(metadata_path)
    actual_hash = sha256_file(volume_path)
    if actual_hash != metadata.source_sha256:
        raise ValueError("Local file SHA-256 does not match authoritative source_sha256; refusing registration")
    record: dict[str, Any] = {
        "schema_version": "1.0", "kind": "volume-registration", "dataset_id": metadata.dataset_id,
        "artifact": {"path": str(volume_path.resolve()), "size_bytes": volume_path.stat().st_size, "sha256": actual_hash},
        "metadata": as_json(metadata), "operations": [],
        "reproducibility": {"code": "vigilia-connectome", "operation": "register-local-volume", "deterministic": True},
    }
    destination = output_root / "registrations" / f"{metadata.dataset_id}.json"
    if destination.exists():
        existing = read_json(destination)
        if existing.get("artifact", {}).get("sha256") != actual_hash:
            raise FileExistsError(f"Immutable registration exists with a different checksum: {destination}")
        return destination
    write_json_atomic(destination, record)
    return destination
