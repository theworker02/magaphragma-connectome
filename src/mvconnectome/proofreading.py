"""Append-only proofreading over machine-derived supervoxel graphs.

This is deliberately a *review workspace*, not a biological-reconstruction
writer.  A machine label volume may be useful for an annotator to merge or
split locally, but neither a workspace nor an edit creates ``MV-FRAG`` or
``MV-N`` records.  Promotion remains a separate, evidence and review-gated
operation.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file, write_json_atomic


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def create_workspace(
    *, crop_manifest_path: Path, crop_id: str, supervoxels_path: Path,
    output: Path, source_method: str, source_run_id: str,
) -> dict[str, Any]:
    """Create an immutable description of a machine-only review workspace."""
    manifest = _load(crop_manifest_path)
    crop = next((item for item in manifest["crops"] if item["id"] == crop_id), None)
    if crop is None:
        raise ValueError(f"Unknown annotation crop: {crop_id}")
    if crop["parent_region_id"] == "MV-GTVOL-000004":
        raise ValueError("Regression-only cube cannot be made into a proofreading workspace")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite review workspace {output}")
    labels = np.load(supervoxels_path, mmap_mode="r", allow_pickle=False)
    if labels.ndim != 3 or tuple(labels.shape) != tuple(crop["raw_shape_zyx"]):
        raise ValueError("Supervoxel labels must be a 3-D ZYX array aligned with the immutable raw crop")
    if labels.dtype.kind not in {"u", "i"} or int(labels.min()) < 0:
        raise ValueError("Supervoxel labels must use non-negative integer identifiers")
    identifiers, counts = np.unique(labels, return_counts=True)
    non_background = [(int(identifier), int(count)) for identifier, count in zip(identifiers, counts, strict=True) if identifier != 0]
    if not non_background:
        raise ValueError("A proofreading workspace needs at least one machine supervoxel")
    value = {
        "schema_version": 1,
        "id": f"MV-PROOF-WORKSPACE-{crop_id[-6:]}",
        "created_at": _now(),
        "status": "MACHINE_SUPERVOXEL_GRAPH_REVIEW_REQUIRED",
        "review_state": "UNREVIEWED",
        "crop_id": crop_id,
        "parent_region_id": crop["parent_region_id"],
        "coordinate_frame": crop["coordinate_frame"],
        "source_origin_xyz": crop["source_origin_xyz"],
        "shape_zyx": list(labels.shape),
        "raw": {"path": crop["raw_crop_path"], "sha256": crop["raw_crop_sha256"]},
        "machine_supervoxels": {
            "path": str(supervoxels_path.resolve()), "sha256": sha256_file(supervoxels_path),
            "dtype": str(labels.dtype), "count": len(non_background),
            "background_id": 0, "source_method": source_method, "source_run_id": source_run_id,
        },
        "prohibited_promotions": ["MV-FRAG", "MV-N", "MV-SYN", "MV-CONN", "REVIEWED_DVID_LABEL"],
        "edit_log": {"path": str((output.parent / f"{output.stem}.events.jsonl").resolve()), "sha256": None, "append_only": True},
    }
    write_json_atomic(output, value)
    return value


def append_review_event(workspace_path: Path, *, event: dict[str, Any]) -> dict[str, Any]:
    """Append a reviewer operation without mutating source labels or the workspace.

    ``MERGE`` identifies two or more supervoxels. ``SPLIT`` records reviewer
    seed points in local ZYX coordinates; it does not pretend to generate a
    split automatically. The actual corrected label must still be registered
    through the reviewed-label gate after inspection.
    """
    workspace = _load(workspace_path)
    if workspace.get("status") != "MACHINE_SUPERVOXEL_GRAPH_REVIEW_REQUIRED":
        raise ValueError("Workspace is not eligible for machine-supervoxel review")
    kind = event.get("kind")
    reviewer = str(event.get("reviewer", "")).strip()
    if kind not in {"MERGE", "SPLIT", "FLAG", "REJECT", "SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"} or not reviewer:
        raise ValueError("Review event needs a reviewer and a supported proofreading kind")
    labels = np.load(workspace["machine_supervoxels"]["path"], mmap_mode="r", allow_pickle=False)
    if kind == "MERGE":
        ids = event.get("supervoxel_ids")
        if not isinstance(ids, list) or len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("MERGE requires two or more distinct supervoxel IDs")
        present = set(np.unique(labels).tolist())
        if any(not isinstance(value, int) or value <= 0 or value not in present for value in ids):
            raise ValueError("MERGE references a missing or background supervoxel")
    if kind == "SPLIT":
        seeds = event.get("seed_points_zyx")
        if not isinstance(seeds, list) or len(seeds) < 2:
            raise ValueError("SPLIT requires at least two reviewer seed points")
        shape = labels.shape
        for point in seeds:
            if not isinstance(point, list) or len(point) != 3 or any(not isinstance(v, int) for v in point):
                raise ValueError("SPLIT seeds must be integer local ZYX points")
            if any(v < 0 or v >= limit for v, limit in zip(point, shape, strict=True)):
                raise ValueError("SPLIT seed lies outside the review crop")
    if kind in {"SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN"}:
        candidate = event.get("candidate")
        if not isinstance(candidate, dict) or not isinstance(candidate.get("supervoxel_a"), int) or not isinstance(candidate.get("supervoxel_b"), int):
            raise ValueError("Boundary decisions require the ranked candidate and both supervoxel IDs")
        present = set(np.unique(labels).tolist())
        if candidate["supervoxel_a"] not in present or candidate["supervoxel_b"] not in present:
            raise ValueError("Boundary decision references a missing supervoxel")
    receipt = {
        "id": f"MV-PROOF-EVENT-{int(datetime.now(UTC).timestamp() * 1_000_000)}",
        "timestamp": _now(), "workspace_id": workspace["id"], "kind": kind,
        "reviewer": reviewer, "payload": {key: value for key, value in event.items() if key not in {"kind", "reviewer"}},
        "effect": "REVIEW_DECISION_RECORDED_NOT_BIOLOGICAL_PROMOTION",
    }
    log = Path(workspace["edit_log"]["path"])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, sort_keys=True) + "\n")
    return receipt


def rank_boundary_candidates(
    *, workspace_path: Path, affinity_path: Path | None, output: Path, maximum: int = 250,
    strategy: str = "uncertainty",
) -> dict[str, Any]:
    """Rank local supervoxel interfaces for targeted SAME/DIFFERENT review.

    Scores intentionally rank *uncertainty and possible merge impact*, not a
    biological decision.  Every resulting item remains a question for a human:
    it contains adjacent machine supervoxels, a local ZYX point, and source
    hashes so a review decision can become affinity supervision later.
    """
    if maximum < 1 or strategy not in {"uncertainty", "low_affinity", "raw_boundary"}:
        raise ValueError("maximum must be positive")
    workspace = _load(workspace_path)
    labels = np.load(workspace["machine_supervoxels"]["path"], mmap_mode="r", allow_pickle=False)
    affinities = None
    if strategy != "raw_boundary":
        if affinity_path is None:
            raise ValueError("Affinity ranking requires an aligned CZYX affinity tensor")
        affinities = np.load(affinity_path, mmap_mode="r", allow_pickle=False)
        if affinities.ndim != 4 or affinities.shape[0] != 3 or tuple(affinities.shape[1:]) != tuple(labels.shape):
            raise ValueError("Expected aligned CZYX three-channel affinity output")
    raw = np.load(workspace["raw"]["path"], mmap_mode="r", allow_pickle=False)
    if tuple(raw.shape) != tuple(labels.shape):
        raise ValueError("Workspace raw EM is not aligned with supervoxels")
    low, high = np.percentile(raw, (1, 99)); scale = max(float(high - low), 1.0)
    # Use every face once. Channel order is the already-verified Z/Y/X order.
    edges: dict[tuple[int, int], dict[str, Any]] = {}
    for axis in range(3):
        left = [slice(None)] * 3; right = [slice(None)] * 3
        left[axis] = slice(0, -1); right[axis] = slice(1, None)
        a = labels[tuple(left)]; b = labels[tuple(right)]
        mask = (a > 0) & (b > 0) & (a != b)
        points = np.argwhere(mask)
        values = np.asarray(affinities[axis][tuple(left)])[mask] if affinities is not None else np.full(len(points), np.nan)
        boundaries = np.abs(np.asarray(raw[tuple(left)], dtype=np.float32) - np.asarray(raw[tuple(right)], dtype=np.float32))[mask] / scale
        for point, value, boundary in zip(points, values, boundaries, strict=True):
            identifier_a = int(a[tuple(point)]); identifier_b = int(b[tuple(point)])
            key = tuple(sorted((identifier_a, identifier_b)))
            entry = edges.setdefault(key, {"channels": [[], [], []], "boundaries": [], "points_zyx": []})
            local = [int(v) for v in point]
            local[axis] += 0  # point is the first voxel of the directed neighbor pair.
            entry["channels"][axis].append(float(value)); entry["boundaries"].append(float(boundary)); entry["points_zyx"].append(local)
    candidates = []
    for ordinal, ((first, second), entry) in enumerate(edges.items(), 1):
        evidence = [value for channel in entry["channels"] for value in channel if not np.isnan(value)]
        mean_affinity = float(np.mean(evidence)) if evidence else None
        mean_boundary = float(np.mean(entry["boundaries"]))
        # Most informative model-side interfaces have ambiguous predicted
        # affinity and substantial contact, neither of which is a human label.
        uncertainty = 1.0 - min(1.0, abs(mean_affinity - 0.5) * 2.0) if mean_affinity is not None else None
        contact = len(entry["points_zyx"])
        score = uncertainty * np.log1p(contact) if strategy == "uncertainty" else (1.0 - mean_affinity) * np.log1p(contact) if strategy == "low_affinity" else mean_boundary * np.log1p(contact)
        point = entry["points_zyx"][len(entry["points_zyx"]) // 2]
        candidates.append({
            "id": f"MV-PROOF-CAND-{ordinal:08d}", "supervoxel_a": first, "supervoxel_b": second,
            "local_point_zyx": point, "interface_voxel_count": contact,
            "affinity_mean": mean_affinity, "affinity_min": float(min(evidence)) if evidence else None, "affinity_max": float(max(evidence)) if evidence else None,
            "raw_boundary_mean": mean_boundary,
            "uncertainty_score": uncertainty, "priority_score": float(score),
            "question": "Do these adjacent regions belong to the SAME neuronal process or DIFFERENT neuronal processes?",
            "selection_strategy": strategy,
            "status": "PROOFREADING_CANDIDATE_MACHINE_RANKED_NOT_A_DECISION",
        })
    candidates.sort(key=lambda item: item["priority_score"], reverse=True)
    result = {
        "kind": "DVID_PROOFREADING_ACTIVE_LEARNING_QUEUE_V1", "created_at": _now(),
        "workspace_id": workspace["id"], "raw_sha256": workspace["raw"]["sha256"],
        "supervoxel_sha256": workspace["machine_supervoxels"]["sha256"], "affinity_sha256": sha256_file(affinity_path) if affinity_path else None,
        "selection_policy": "rank ambiguous interfaces by contact extent" if strategy == "uncertainty" else "rank low-affinity interfaces as possible separation counterevidence" if strategy == "low_affinity" else "rank raw-EM intensity discontinuities; humans determine biological separation",
        "candidates": candidates[:maximum], "total_interfaces": len(candidates),
        "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
    }
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite active-learning queue {output}")
    write_json_atomic(output, result)
    return {"candidates": len(result["candidates"]), "total_interfaces": len(candidates), "status": "REVIEW_REQUIRED"}
