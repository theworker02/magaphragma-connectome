"""Reproducible adapter for Google Flood-Filling Networks (FFN).

This module intentionally does not vendor, import, or modify the legacy FFN
TensorFlow codebase.  It prepares registered CATMAID morphology as held-out
validation material, invokes a separately pinned FFN checkout, and records
only machine-only candidate evidence.  A passing validation gate is required
before an FFN result can be used to propose unresolved extensions.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from .io import sha256_file, write_json_atomic


FFN_REPOSITORY = "https://github.com/google/ffn"
FFN_LICENSE = "Apache-2.0"
FFN_ADAPTER_VERSION = "1.0"


def _timestamp() -> str:
    """Return a UTC receipt timestamp for externally executed FFN work."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    """Load a manifest-like JSON object used by explicit adapter gates."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_nodes(path: Path) -> list[dict[str, Any]]:
    """Read CATMAID morphology rows without silently accepting unknown formats."""
    return pq.read_table(path).to_pylist()


def _matrix(value: Any) -> np.ndarray:
    """Validate the registration matrix used to bridge CATMAID and volume frames."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("Registration matrix must be a 4 by 4 source-nm to target-voxel matrix")
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-14:
        raise ValueError("Registration matrix must be finite and invertible")
    return matrix


@dataclass(frozen=True, slots=True)
class VerifiedRegistration:
    """An explicit, independently validated CATMAID-to-EM transform.

    The project currently has no such transform.  Requiring this record makes
    it impossible to silently treat CATMAID coordinates as FIB-SEM voxels.
    """

    id: str
    source_frame_id: str
    target_volume_id: str
    status: str
    source_nm_to_target_voxel_matrix: tuple[tuple[float, float, float, float], ...]
    validation_method: str
    validation_evidence_ids: tuple[str, ...]

    @classmethod
    def from_path(cls, path: Path) -> "VerifiedRegistration":
        value = _load_json(path)
        matrix = _matrix(value.get("source_nm_to_target_voxel_matrix"))
        result = cls(
            id=str(value.get("id", "")), source_frame_id=str(value.get("source_frame_id", "")),
            target_volume_id=str(value.get("target_volume_id", "")), status=str(value.get("status", "")),
            source_nm_to_target_voxel_matrix=tuple(tuple(float(x) for x in row) for row in matrix),
            validation_method=str(value.get("validation_method", "")),
            validation_evidence_ids=tuple(str(item) for item in value.get("validation_evidence_ids", [])),
        )
        if not result.id.startswith("MV-MAP-"):
            raise ValueError("Registration id must begin with MV-MAP-")
        if result.status != "VERIFIED" or not result.validation_method or not result.validation_evidence_ids:
            raise ValueError("FFN seed export requires a VERIFIED registration with validation evidence")
        return result

    def transform(self, point_nm_xyz: Iterable[float]) -> tuple[float, float, float]:
        point = np.asarray([*point_nm_xyz, 1.0], dtype=float)
        output = np.asarray(self.source_nm_to_target_voxel_matrix) @ point
        if abs(output[3]) < 1e-14:
            raise ValueError("Registration generated an invalid homogeneous coordinate")
        return tuple(float(x) for x in output[:3] / output[3])


def plan_subvolumes(ingestion_path: Path, output: Path, chunk_xyz: tuple[int, int, int] = (256, 256, 64), halo_xyz: tuple[int, int, int] = (32, 32, 16)) -> Path:
    """Produce deterministic, halo-aware FFN work units for one ingested crop."""
    ingestion = _load_json(ingestion_path)
    shape_zyx = tuple(int(item) for item in np.load(ingestion["raw_array"], mmap_mode="r").shape)
    shape_xyz = (shape_zyx[2], shape_zyx[1], shape_zyx[0])
    if any(value <= 0 for value in chunk_xyz + halo_xyz):
        raise ValueError("Chunk and halo dimensions must be positive")
    if any(halo >= chunk for halo, chunk in zip(halo_xyz, chunk_xyz)):
        raise ValueError("Each halo must be smaller than its chunk dimension")
    origin = ingestion["source_bounds_xyz"]
    origin_xyz = (int(origin["x"][0]), int(origin["y"][0]), int(origin["z"][0]))
    units: list[dict[str, Any]] = []
    for z in range(0, shape_xyz[2], chunk_xyz[2]):
        for y in range(0, shape_xyz[1], chunk_xyz[1]):
            for x in range(0, shape_xyz[0], chunk_xyz[0]):
                start = (x, y, z); stop = tuple(min(start[i] + chunk_xyz[i], shape_xyz[i]) for i in range(3))
                expanded_start = tuple(max(0, start[i] - halo_xyz[i]) for i in range(3))
                expanded_stop = tuple(min(shape_xyz[i], stop[i] + halo_xyz[i]) for i in range(3))
                units.append({
                    "id": f"MV-FFN-WU-{len(units)+1:06d}", "core_local_voxel_bounds_xyz": [[start[i], stop[i]] for i in range(3)],
                    "read_local_voxel_bounds_xyz": [[expanded_start[i], expanded_stop[i]] for i in range(3)],
                    "core_global_voxel_bounds_xyz": [[origin_xyz[i] + start[i], origin_xyz[i] + stop[i]] for i in range(3)],
                })
    plan = {
        "kind": "FFN_SUBVOLUME_PLAN", "adapter_version": FFN_ADAPTER_VERSION, "created_at": _timestamp(),
        "ingestion": {"path": str(ingestion_path), "raw_sha256": ingestion["raw_sha256"], "volume_id": ingestion["volume_id"], "region_id": ingestion["region_id"]},
        "shape_local_voxel_xyz": list(shape_xyz), "chunk_xyz": list(chunk_xyz), "halo_xyz": list(halo_xyz), "work_units": units,
        "infrastructure_basis": "Google Research Connectomics-style explicit bounding boxes and subvolume planning; no external package is required for this stable local contract.",
    }
    output.parent.mkdir(parents=True, exist_ok=True); write_json_atomic(output, plan)
    return output


def build_seed_and_holdout_manifests(nodes_path: Path, physical_neurons_path: Path, registration_path: Path, ingestion_path: Path, output: Path, holdout_fraction: float = 0.2) -> dict[str, Path]:
    """Turn registered CATMAID endpoints into FFN seeds and sparse validation paths."""
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    registration = VerifiedRegistration.from_path(registration_path)
    ingestion = _load_json(ingestion_path)
    if registration.target_volume_id != ingestion["volume_id"]:
        raise ValueError("Registration target volume does not match FFN ingestion volume")
    physical = _load_json(physical_neurons_path)
    frame_id = physical["coordinate_frame"]["id"]
    if registration.source_frame_id != frame_id:
        raise ValueError("Registration source frame does not match physical CATMAID morphology frame")
    nodes = _read_nodes(nodes_path)
    by_neuron: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        by_neuron.setdefault(node["mv_neuron_id"], []).append(node)
    origin = ingestion["source_bounds_xyz"]
    origin_xyz = (int(origin["x"][0]), int(origin["y"][0]), int(origin["z"][0]))
    raw_shape_zyx = tuple(int(item) for item in np.load(ingestion["raw_array"], mmap_mode="r").shape)
    raw_shape_xyz = (raw_shape_zyx[2], raw_shape_zyx[1], raw_shape_zyx[0])
    all_records: list[dict[str, Any]] = []
    for neuron_id, neuron_nodes in sorted(by_neuron.items()):
        parents = {node["parent_treenode_id"] for node in neuron_nodes if node["parent_treenode_id"] is not None}
        endpoints = [node for node in neuron_nodes if node["source_treenode_id"] not in parents or node["parent_treenode_id"] is None]
        references = []
        for node in neuron_nodes:
            global_voxel = registration.transform((node["x_nm"], node["y_nm"], node["z_nm"]))
            local = tuple(round(global_voxel[i] - origin_xyz[i]) for i in range(3))
            if all(0 <= local[i] < raw_shape_xyz[i] for i in range(3)):
                references.append([local[2], local[1], local[0]])  # numpy volume order zyx
        seed_points = []
        for node in endpoints:
            global_voxel = registration.transform((node["x_nm"], node["y_nm"], node["z_nm"]))
            local = tuple(round(global_voxel[i] - origin_xyz[i]) for i in range(3))
            if all(0 <= local[i] < raw_shape_xyz[i] for i in range(3)):
                seed_points.append({"source_treenode_id": node["source_treenode_id"], "local_voxel_zyx": [local[2], local[1], local[0]]})
        if references and seed_points:
            all_records.append({"mv_neuron_id": neuron_id, "seed_points": seed_points, "reference_path_voxels_zyx": references})
    holdout = [record for record in all_records if int(hashlib.sha256(record["mv_neuron_id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < holdout_fraction]
    training = [record for record in all_records if record not in holdout]
    common = {"kind": "FFN_REGISTERED_CATMAID_SEEDS", "adapter_version": FFN_ADAPTER_VERSION, "created_at": _timestamp(), "release_eligible": False,
              "registration": {"id": registration.id, "sha256": sha256_file(registration_path), "source_frame_id": registration.source_frame_id, "target_volume_id": registration.target_volume_id},
              "ingestion": {"path": str(ingestion_path), "raw_sha256": ingestion["raw_sha256"], "local_voxel_origin_xyz": list(origin_xyz), "shape_local_voxel_zyx": list(raw_shape_zyx)},
              "provenance": "CATMAID endpoints and morphology are seed/validation material only; they do not establish a new biological identity or release eligibility."}
    output.mkdir(parents=True, exist_ok=True)
    seeds_path = output / "ffn-seeds.json"; holdout_path = output / "ffn-heldout-morphology.json"
    write_json_atomic(seeds_path, {**common, "partition": "TRAINING", "records": training})
    write_json_atomic(holdout_path, {**common, "partition": "HELD_OUT", "records": holdout})
    return {"seeds": seeds_path, "holdout": holdout_path}


def run_ffn(ffn_checkout: Path, inference_request: Path, bounding_box: str, output: Path, python_executable: str = sys.executable) -> Path:
    """Invoke a pinned external FFN checkout and write a tamper-evident run receipt."""
    script = ffn_checkout / "run_inference.py"
    if not script.is_file():
        raise ValueError("--ffn-checkout must contain Google's run_inference.py")
    if not inference_request.is_file():
        raise ValueError("FFN inference request does not exist")
    if not bounding_box.strip() or any(character not in "0123456789{}:, xystartsize-" for character in bounding_box):
        raise ValueError("Bounding box must use FFN's explicit start/size syntax")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    command = [python_executable, str(script), "--inference_request", str(inference_request), "--bounding_box", bounding_box]
    completed = subprocess.run(command, cwd=ffn_checkout, capture_output=True, text=True, check=False, timeout=60 * 60)
    produced = sorted(output.rglob("*.npz"))
    receipt = {"kind": "FFN_INFERENCE_RUN", "adapter_version": FFN_ADAPTER_VERSION, "started_at": _timestamp(), "runtime_seconds": time.time() - started,
               "ffn": {"repository": FFN_REPOSITORY, "license": FFN_LICENSE, "checkout": str(ffn_checkout), "run_script_sha256": sha256_file(script)},
               "inference_request": {"path": str(inference_request), "sha256": sha256_file(inference_request)}, "command": command,
               "bounding_box": bounding_box, "returncode": completed.returncode, "stdout": completed.stdout[-8000:], "stderr": completed.stderr[-8000:],
               "outputs": [{"path": str(item), "sha256": sha256_file(item)} for item in produced],
               "review_state": "MACHINE_ONLY", "release_eligible": False}
    receipt_path = output / "ffn-run.json"; write_json_atomic(receipt_path, receipt)
    if completed.returncode:
        raise RuntimeError(f"FFN inference failed; receipt retained at {receipt_path}")
    return receipt_path


def import_ffn_labels(source: Path, output: Path) -> Path:
    """Normalize a 3-D FFN NPZ result into the project's explicit ZYX NPY contract."""
    with np.load(source) as archive:
        preferred = ("segmentation", "labels", "seg")
        key = next((name for name in preferred if name in archive and archive[name].ndim == 3), None)
        if key is None:
            candidates = [name for name in archive.files if archive[name].ndim == 3]
            if len(candidates) != 1:
                raise ValueError("Could not identify one 3-D segmentation array in FFN NPZ output")
            key = candidates[0]
        labels = np.asarray(archive[key])
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("FFN segmentation labels must use an integer dtype")
    output.parent.mkdir(parents=True, exist_ok=True); np.save(output, labels)
    receipt = output.with_suffix(".import.json")
    write_json_atomic(receipt, {"kind": "FFN_LABEL_IMPORT", "created_at": _timestamp(), "source": {"path": str(source), "sha256": sha256_file(source), "array_key": key},
                                "output": {"path": str(output), "sha256": sha256_file(output), "shape_zyx": list(labels.shape), "dtype": str(labels.dtype)}, "release_eligible": False})
    return output


def validate_ffn_run(labels_path: Path, heldout_path: Path, output: Path, minimum_path_coverage: float = 0.8, maximum_merge_pairs: int = 0) -> Path:
    """Measure sparse CATMAID-path coverage, splits, and label-sharing merges."""
    if not 0 <= minimum_path_coverage <= 1 or maximum_merge_pairs < 0:
        raise ValueError("Invalid FFN validation thresholds")
    labels = np.load(labels_path, mmap_mode="r")
    if labels.ndim != 3:
        raise ValueError("FFN label input must be a 3D zyx numpy array")
    heldout = _load_json(heldout_path); rows: list[dict[str, Any]] = []
    label_owners: dict[int, set[str]] = {}
    for record in heldout["records"]:
        sample_labels: list[int] = []
        for z, y, x in record["reference_path_voxels_zyx"]:
            if 0 <= z < labels.shape[0] and 0 <= y < labels.shape[1] and 0 <= x < labels.shape[2]:
                value = int(labels[z, y, x])
                if value > 0: sample_labels.append(value)
        counts = {label: sample_labels.count(label) for label in set(sample_labels)}
        dominant = max(counts, key=counts.get) if counts else None
        coverage = (counts[dominant] / len(record["reference_path_voxels_zyx"])) if dominant is not None else 0.0
        for label in counts: label_owners.setdefault(label, set()).add(record["mv_neuron_id"])
        rows.append({"mv_neuron_id": record["mv_neuron_id"], "reference_node_count": len(record["reference_path_voxels_zyx"]), "dominant_ffn_label": dominant,
                     "path_coverage": coverage, "labels_on_path": len(counts), "status": "PASS" if coverage >= minimum_path_coverage else "FAIL"})
    merge_pairs = sum(len(owners) * (len(owners) - 1) // 2 for owners in label_owners.values() if len(owners) > 1)
    mean_coverage = sum(item["path_coverage"] for item in rows) / len(rows) if rows else 0.0
    gate = "PASSED" if rows and all(item["status"] == "PASS" for item in rows) and merge_pairs <= maximum_merge_pairs else "BLOCKED"
    result = {"kind": "FFN_HELD_OUT_VALIDATION", "created_at": _timestamp(), "release_eligible": False, "labels": {"path": str(labels_path), "sha256": sha256_file(labels_path), "shape_zyx": list(labels.shape)},
              "heldout": {"path": str(heldout_path), "sha256": sha256_file(heldout_path), "registration": heldout["registration"]},
              "thresholds": {"minimum_path_coverage": minimum_path_coverage, "maximum_merge_pairs": maximum_merge_pairs},
              "metrics": {"neuron_count": len(rows), "mean_path_coverage": mean_coverage, "merge_pairs": merge_pairs}, "records": rows,
              "candidate_extension_gate": gate, "policy": "Only a PASSED gate permits downstream candidate-extension triage. It never promotes a candidate to a biological reconstruction."}
    output.parent.mkdir(parents=True, exist_ok=True); write_json_atomic(output, result)
    return output
