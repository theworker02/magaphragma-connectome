"""Conservative, explicitly non-review automatic DVID consensus labels."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import ndimage

from .annotation_crops import IGNORE_LABEL
from .io import sha256_file, write_json_atomic


def extract_parent_affinity_crop(crop_manifest_path: Path, crop_id: str, parent_affinity_path: Path, output: Path) -> dict:
    """Extract an aligned affinity crop from a retained parent inference output."""
    manifest = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
    crop = next((item for item in manifest["crops"] if item["id"] == crop_id), None)
    if crop is None:
        raise ValueError("Unknown crop")
    affinity = np.load(parent_affinity_path, mmap_mode="r", allow_pickle=False)
    origin, shape = tuple(crop["origin_zyx"]), tuple(crop["shape_zyx"])
    if affinity.ndim != 4 or affinity.shape[0] != 3:
        raise ValueError("Expected parent affinity in CZYX")
    source = (slice(None),) + tuple(slice(start, start + size) for start, size in zip(origin, shape, strict=True))
    value = np.asarray(affinity[source]).copy()
    if value.shape != (3,) + shape:
        raise ValueError("Parent affinity does not cover crop")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    np.save(output, value, allow_pickle=False)
    return {"crop_id": crop_id, "status": "MACHINE_PSEUDOLABEL", "affinity_sha256": sha256_file(output), "shape_czyx": list(value.shape), "parent_affinity_sha256": sha256_file(parent_affinity_path)}


def build_auto_consensus(crop_manifest_path: Path, crop_id: str, affinity_path: Path, output_dir: Path) -> dict:
    """Intersect model interior with raw-image boundary support, fail closed.

    This is an experimental pseudo-label generator. The model affinity and its
    watershed descendants have shared ancestry, so only raw-gradient support is
    a separately weighted signal; neither constitutes biological evidence.
    """
    manifest = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
    crop = next((item for item in manifest["crops"] if item["id"] == crop_id), None)
    if crop is None or crop["parent_region_id"] == "MV-GTVOL-000004":
        raise ValueError("Unknown or regression-only crop")
    if crop["split"] not in {"DVID_TARGET_TRAIN", "DVID_TARGET_VALIDATION"}:
        raise ValueError("Crop is not assigned to the auto-consensus branch")
    raw = np.load(crop["raw_crop_path"], allow_pickle=False)
    affinity = np.load(affinity_path, allow_pickle=False)
    if affinity.shape != (3,) + raw.shape or not np.isfinite(affinity).all() or affinity.min() < 0 or affinity.max() > 1:
        raise ValueError("Expected finite three-channel CZYX probabilities aligned with raw crop")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    # Predeclared conservative evidence rule. It is not calibrated against 000004.
    gradient = ndimage.gaussian_gradient_magnitude(raw.astype("float32"), sigma=1.0)
    denominator = float(np.percentile(gradient, 99.5))
    if denominator <= 0:
        raise ValueError("Raw image has no usable local gradient variation")
    boundary = np.clip(gradient / denominator, 0, 1).astype("float32")
    interior = affinity.mean(axis=0)
    high_confidence = (interior >= 0.80) & (boundary <= 0.35)
    components, count = ndimage.label(high_confidence, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(components.ravel())
    keep = sizes >= 128
    keep[0] = False
    components = components * keep[components]
    components, _ = ndimage.label(components > 0, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(components.ravel())[1:]
    labels = np.full(raw.shape, IGNORE_LABEL, dtype=np.uint32)
    labels[components > 0] = components[components > 0]
    near_full = bool(len(sizes) and sizes.max() / raw.size >= 0.5)
    too_empty = int((components > 0).sum()) < 4096
    pathological_count = int(components.max()) > 10000
    # A single retained component supplies only same-process pairs. It cannot
    # teach the model a target-domain separation boundary, so it is not enough
    # to admit the crop into experimental adaptation supervision.
    insufficient_relationship_diversity = int(components.max()) < 2
    accepted = bool(components.max()) and not near_full and not too_empty and not pathological_count and not insufficient_relationship_diversity
    status = "AUTO_CONSENSUS_LABEL" if accepted else "REJECTED_AUTO_CONSENSUS"
    np.save(output_dir / "labels_zyx.npy", labels, allow_pickle=False)
    np.save(output_dir / "raw_gradient_boundary_zyx.npy", boundary, allow_pickle=False)
    np.save(output_dir / "high_confidence_mask_zyx.npy", high_confidence, allow_pickle=False)
    record = {
        "id": f"MV-AUTOANN-{crop_id[-6:]}", "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status, "classification": "EXPERIMENTAL_SUPERVISION_NOT_REVIEWED_DVID_LABEL",
        "crop_id": crop_id, "split": "AUTO_CONSENSUS_TARGET_TRAIN" if crop["split"] == "DVID_TARGET_TRAIN" else "AUTO_CONSENSUS_TARGET_VALIDATION",
        "raw": {"sha256": crop["raw_crop_sha256"], "shape_zyx": list(raw.shape), "source_origin_xyz": crop["source_origin_xyz"]},
        "evidence_ancestry": [
            {"method": "SegNeuron step-2250 affinity", "path": str(affinity_path.resolve()), "sha256": sha256_file(affinity_path), "independent_vote": False},
            {"method": "raw-EM Gaussian-gradient boundary support", "input_sha256": crop["raw_crop_sha256"], "sigma_voxels": 1.0, "independent_vote": True}
        ],
        "consensus_rule": {"model_mean_affinity_min": 0.80, "raw_gradient_boundary_max": 0.35, "minimum_component_voxels": 128, "uncertain_encoding": IGNORE_LABEL},
        "outputs": {"labels": "labels_zyx.npy", "labels_sha256": sha256_file(output_dir / "labels_zyx.npy"), "boundary": "raw_gradient_boundary_zyx.npy", "high_confidence": "high_confidence_mask_zyx.npy"},
        "statistics": {"high_confidence_voxels": int((components > 0).sum()), "ignore_voxels": int((labels == IGNORE_LABEL).sum()), "instance_count": int(components.max()), "largest_component_fraction": float(sizes.max() / raw.size) if len(sizes) else 0.0},
        "sanity": {"near_full_volume_component": near_full, "effectively_empty": too_empty, "pathological_component_count": pathological_count, "insufficient_same_different_relationship_diversity": insufficient_relationship_diversity, "accepted_for_experimental_supervision": accepted},
        "prohibited_uses": ["reviewed_dvid_label", "biological_evidence", "mv_frag", "mv_neuron", "mv_conn"]
    }
    write_json_atomic(output_dir / "record.json", record)
    return record
