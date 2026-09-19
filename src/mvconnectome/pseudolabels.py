"""Strictly non-ground-truth pseudolabel generation and automated review ranking."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .io import sha256_file, write_json_atomic


def generate_interior_component_proposals(inference_dir: Path, raw_receipt: Path, output: Path, qa_output: Path, *, interior_threshold: float = 0.5, minimum_voxels: int = 64) -> dict:
    """Make proposal-only components from retained SegNeuron affinities.

    This is deliberately not FRMC and never claims neuronal reconstruction.
    Thresholds are predeclared proposal heuristics; the artifacts retain model
    probabilities so a reviewer can reject, split, or merge every proposal.
    """
    import numpy as np
    from scipy import ndimage

    if not 0.0 < interior_threshold < 1.0 or minimum_voxels < 1:
        raise ValueError("Invalid pseudolabel proposal parameters")
    run = json.loads((inference_dir / "inference.json").read_text(encoding="utf-8"))
    receipt = json.loads(raw_receipt.read_text(encoding="utf-8"))
    affinities = np.load(inference_dir / "affinities.npy", allow_pickle=False)
    if affinities.ndim != 4 or affinities.shape[0] != 3:
        raise ValueError("Expected retained SegNeuron affinities in CZYX with three neighbor offsets")
    # This is deliberately not the upstream FRMC instance postprocessor.
    interior = affinities.mean(axis=0)
    labels, count = ndimage.label(interior >= interior_threshold, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.ravel())
    retained = sizes >= minimum_voxels
    retained[0] = False
    labels = labels * retained[labels]
    labels, _ = ndimage.label(labels > 0, structure=ndimage.generate_binary_structure(3, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"Refusing to overwrite pseudolabel proposal {output}")
    np.save(output, labels.astype(np.uint32), allow_pickle=False)

    component_sizes = np.bincount(labels.ravel())[1:]
    uncertainty = ((affinities >= 0.4) & (affinities <= 0.6)).mean(axis=0)
    flags = []
    for identifier, size in enumerate(component_sizes, 1):
        mask = labels == identifier
        points = np.argwhere(mask)
        boundary = any((points[:, axis] == 0).any() or (points[:, axis] == labels.shape[axis] - 1).any() for axis in range(3))
        item_flags = []
        if size < 512:
            item_flags.append("TINY_FRAGMENT")
        if size / labels.size >= 0.5:
            item_flags.append("NEAR_FULL_VOLUME_MERGE_CANDIDATE")
        if boundary:
            item_flags.append("VOLUME_BOUNDARY")
        if float(uncertainty[mask].mean()) >= 0.25:
            item_flags.append("HIGH_AFFINITY_UNCERTAINTY")
        flags.append({"proposal_label": identifier, "voxel_count": int(size), "flags": item_flags, "mean_affinity_uncertainty": float(uncertainty[mask].mean())})
    proposal = {
        "id": "MV-PSEUDOLABEL-SEGNEURON-0001",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "MACHINE_PSEUDOLABEL",
        "review_state": "REVIEW_REQUIRED",
        "source_input": receipt,
        "model_run": {"path": str((inference_dir / "inference.json").resolve()), "checkpoint_sha256": run["checkpoint"]["sha256"], "model": run["model"]},
        "method": {"name": "AFFINITY_COMPONENT_PROPOSAL_V0", "mean_affinity_threshold": interior_threshold, "minimum_voxels": minimum_voxels, "not_a_production_instance_segmenter": True},
        "proposal_path": str(output.resolve()), "proposal_sha256": sha256_file(output),
        "proposal_instances": int(labels.max()), "proposal_labeled_voxels": int((labels > 0).sum()),
        "raw_model_outputs_retained": True,
        "prohibited_uses": ["ground_truth", "held_out_metric_target", "production_mv_seg", "biological_neuron_identity"],
    }
    write_json_atomic(qa_output, {"proposal": proposal, "component_flags": flags, "model_disagreement": "NOT_AVAILABLE_SINGLE_MODEL", "automated_repair": "NOT_RUN_REQUIRES_REVIEW_POLICY"})
    return {"proposal_instances": proposal["proposal_instances"], "proposal_labeled_voxels": proposal["proposal_labeled_voxels"], "qa_items": len(flags), "status": proposal["status"]}


def audit_existing_proposal(proposal_path: Path, qa_output: Path) -> dict:
    """Append-only severity audit for an existing machine pseudolabel artifact."""
    import numpy as np

    labels = np.load(proposal_path, allow_pickle=False)
    if labels.ndim != 3 or labels.dtype.kind not in {"u", "i"}:
        raise ValueError("Proposal must be a three-dimensional integer label volume")
    sizes = np.bincount(labels.ravel())[1:]
    findings = [{"proposal_label": index, "voxel_count": int(size), "flags": ["NEAR_FULL_VOLUME_MERGE_CANDIDATE"] if size / labels.size >= 0.5 else []} for index, size in enumerate(sizes, 1)]
    result = {"kind": "PSEUDOLABEL_SEVERITY_AUDIT", "proposal_path": str(proposal_path.resolve()), "proposal_sha256": sha256_file(proposal_path), "findings": findings, "automatic_decision": "REVIEW_REQUIRED_NOT_ELIGIBLE_FOR_GROUND_TRUTH_OR_PRODUCTION"}
    write_json_atomic(qa_output, result)
    return {"components": len(findings), "near_full_volume_merge_candidates": sum(bool(item["flags"]) for item in findings)}
