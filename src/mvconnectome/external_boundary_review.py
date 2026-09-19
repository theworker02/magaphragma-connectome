"""Independent expert boundary decisions for DVID affinity supervision.

The external reviewer supplies local same/different judgments on raw EM.  This
module deliberately stores no inferred neurons, segments, or biological graph
records.  Its only derived product is a masked CZYX affinity target tensor.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file, write_json_atomic

_DECISIONS = {"SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"}
_CHANNELS = {0: "Z", 1: "Y", 2: "X"}
_PLACEHOLDER_REVIEWERS = {"expert-identifier", "reviewer", "unknown", "tbd", "to-be-assigned"}


def _now() -> str:
    """Return a UTC timestamp for append-only review provenance."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json(path: Path) -> dict[str, Any]:
    """Load one workspace JSON object and reject non-object review artifacts."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def create_external_review_workspace(*, request_path: Path, crop_manifest_path: Path, crop_id: str, output: Path) -> dict[str, Any]:
    """Freeze a rights-aware, raw-only review workspace for one requested crop."""
    request, manifest = _json(request_path), _json(crop_manifest_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite review workspace {output}")
    requested = next((item for item in request.get("requested_review_sets", []) if item.get("crop_id") == crop_id), None)
    crop = next((item for item in manifest.get("crops", []) if item.get("id") == crop_id), None)
    if requested is None or crop is None:
        raise ValueError("Crop must be explicitly listed in the external-review request and crop manifest")
    if crop["parent_region_id"] == "MV-GTVOL-000004" or requested["parent_region_id"] == "MV-GTVOL-000004":
        raise ValueError("Regression-only volume cannot enter external review")
    for key in ("parent_region_id", "raw_crop_sha256", "coordinate_frame", "source_origin_xyz"):
        expected = requested["raw_sha256"] if key == "raw_crop_sha256" else requested[key]
        if crop[key] != expected:
            raise ValueError(f"Request and crop manifest disagree on {key}")
    raw = Path(crop["raw_crop_path"])
    if not raw.exists() or sha256_file(raw) != requested["raw_sha256"]:
        raise ValueError("Immutable raw crop is missing or does not match the review request hash")
    value = {
        "schema_version": 1,
        "id": f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id[-6:]}",
        "created_at": _now(),
        "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
        "review_state": "UNREVIEWED",
        "request": {"path": str(request_path.resolve()), "sha256": sha256_file(request_path), "id": request["id"]},
        "crop_id": crop_id,
        "split": requested["role"],
        "parent_region_id": crop["parent_region_id"],
        "coordinate_frame": crop["coordinate_frame"],
        "source_origin_xyz": crop["source_origin_xyz"],
        "raw": {"path": str(raw.resolve()), "sha256": requested["raw_sha256"], "shape_zyx": crop["raw_shape_zyx"], "dtype": crop["raw_dtype"]},
        "allowed_decisions": sorted(_DECISIONS),
        "pair_contract": "pair_left_zyx is followed by its positive neighbour along channel_zyx: 0=Z, 1=Y, 2=X.",
        "event_log": {"path": str((output.parent / f"{output.stem}.events.jsonl").resolve()), "append_only": True},
        "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
    }
    write_json_atomic(output, value)
    return value


def create_external_scan_queue(*, workspace_path: Path, output: Path, count: int = 72, border: int = 8) -> dict[str, Any]:
    """Freeze spatially stratified navigation centres for human raw-EM review.

    These are deliberately *not* membrane candidates: no raw feature, model,
    segmentation, or earlier decision determines their selection. The reviewer
    still chooses whether a nearby adjacent pair is suitable for a decision.
    """
    workspace = _json(workspace_path)
    if workspace.get("status") != "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED":
        raise ValueError("Expected external raw-EM review workspace")
    if count < 1 or border < 0:
        raise ValueError("count must be positive and border non-negative")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen scan queue {output}")
    shape = tuple(int(value) for value in workspace["raw"]["shape_zyx"])
    if any(limit <= 2 * border + 2 for limit in shape):
        raise ValueError("Crop is too small for the requested review border")
    # Select a near-cubic grid and truncate deterministically. This controls
    # coverage, not candidate quality.
    dimensions = [max(1, round(count ** (1 / 3) * limit / np.cbrt(np.prod(shape)))) for limit in shape]
    while int(np.prod(dimensions)) < count:
        dimensions[int(np.argmax(np.asarray(shape) / np.asarray(dimensions)))] += 1
    axes = [np.rint(np.linspace(border, limit - border - 1, cells)).astype(int) for limit, cells in zip(shape, dimensions, strict=True)]
    points = [[int(z), int(y), int(x)] for z in axes[0] for y in axes[1] for x in axes[2]][:count]
    value = {
        "schema_version": 1, "id": f"MV-EXTERNAL-RAW-SCAN-{workspace['crop_id'][-6:]}", "created_at": _now(),
        "status": "RAW_EM_NAVIGATION_ONLY_REVIEW_REQUIRED", "workspace_id": workspace["id"], "crop_id": workspace["crop_id"],
        "raw_sha256": workspace["raw"]["sha256"], "workspace_sha256": sha256_file(workspace_path),
        "selection": {"method": "SPATIALLY_STRATIFIED_GRID_V1", "count": len(points), "requested_count": count, "border_voxels": border,
                      "prohibited_inputs": ["SegNeuron", "supervoxels", "pretrained_models", "raw_feature_ranking", "prior_review_decisions"]},
        "locations_zyx": points,
        "note": "Locations are navigation aids only. They are not membrane candidates and cannot create training evidence without an explicit reviewer decision.",
    }
    write_json_atomic(output, value)
    return value


def append_external_boundary_event(
    workspace_path: Path, *, reviewer: str, decision: str, pair_left_zyx: list[int], channel_zyx: int,
    rationale: str | None = None, orthogonal_views_inspected: bool = False, question_reference: str | None = None,
    interface_id: str | None = None,
) -> dict[str, Any]:
    """Append a single expert decision after validating its direct affinity geometry."""
    workspace = _json(workspace_path)
    if workspace.get("status") != "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED":
        raise ValueError("Workspace is not eligible for external raw-EM review")
    reviewer = reviewer.strip()
    if not reviewer or reviewer.casefold() in _PLACEHOLDER_REVIEWERS or decision not in _DECISIONS:
        raise ValueError("A reviewer and supported decision are required")
    if not orthogonal_views_inspected:
        raise ValueError("Reviewer must confirm XY, XZ, YZ, and nearby-slice inspection")
    if channel_zyx not in _CHANNELS or len(pair_left_zyx) != 3 or any(not isinstance(value, int) for value in pair_left_zyx):
        raise ValueError("Expected integer pair_left_zyx and channel_zyx in 0..2")
    shape = tuple(workspace["raw"]["shape_zyx"])
    right = list(pair_left_zyx); right[channel_zyx] += 1
    if any(value < 0 or value >= limit for value, limit in zip(pair_left_zyx, shape, strict=True)) or right[channel_zyx] >= shape[channel_zyx]:
        raise ValueError("Adjacent affinity pair lies outside the immutable crop")
    log = Path(workspace["event_log"]["path"])
    key = (tuple(pair_left_zyx), channel_zyx)
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if tuple(event["pair_left_zyx"]) == key[0] and event["channel_zyx"] == key[1]:
                # A historical placeholder entry is preserved for audit but is
                # not qualified evidence. Let an identified reviewer supply
                # the single admissible qualified decision for the same pair.
                if str(event.get("reviewer", "")).strip().casefold() not in _PLACEHOLDER_REVIEWERS:
                    raise ValueError("An append-only qualified decision already exists for this affinity pair")
    event = {
        "id": f"MV-EXTERNAL-BOUNDARY-EVENT-{int(datetime.now(UTC).timestamp() * 1_000_000)}",
        "timestamp": _now(), "workspace_id": workspace["id"], "crop_id": workspace["crop_id"], "split": workspace["split"],
        "raw_sha256": workspace["raw"]["sha256"], "reviewer": reviewer, "decision": decision,
        "pair_left_zyx": pair_left_zyx, "pair_right_zyx": right, "channel_zyx": channel_zyx,
        "channel_name": _CHANNELS[channel_zyx], "orthogonal_views_inspected": True,
        "rationale": rationale.strip() if rationale else None,
        "question_reference": question_reference.strip() if question_reference else None,
        "interface_id": interface_id.strip() if interface_id else None,
        "effect": "REVIEWED_LOCAL_AFFINITY_EVIDENCE_NOT_A_BIOLOGICAL_PROMOTION",
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def materialize_external_boundary_affinity(workspace_path: Path, output: Path) -> dict[str, Any]:
    """Make immutable masked CZYX targets; fail closed without both classes."""
    workspace = _json(workspace_path)
    log = Path(workspace["event_log"]["path"])
    if not log.exists():
        raise ValueError("No external reviewer decisions exist")
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw = np.load(workspace["raw"]["path"], mmap_mode="r", allow_pickle=False)
    if sha256_file(Path(workspace["raw"]["path"])) != workspace["raw"]["sha256"]:
        raise ValueError("Raw source hash changed after review workspace creation")
    shape = tuple(raw.shape)
    targets = np.zeros((3,) + shape, dtype=np.uint8)
    mask = np.zeros((3,) + shape, dtype=np.uint8)
    effective: list[dict[str, Any]] = []
    excluded_placeholder_reviewer_events: list[str] = []
    for event in events:
        if str(event.get("reviewer", "")).strip().casefold() in _PLACEHOLDER_REVIEWERS:
            excluded_placeholder_reviewer_events.append(str(event.get("id")))
            continue
        if event.get("decision") not in {"SAME_PROCESS", "DIFFERENT_PROCESS"}:
            continue
        channel, left = int(event["channel_zyx"]), tuple(event["pair_left_zyx"])
        if mask[(channel,) + left]:
            raise ValueError("Duplicate effective affinity pair in append-only external log")
        targets[(channel,) + left] = 1 if event["decision"] == "SAME_PROCESS" else 0
        mask[(channel,) + left] = 1
        effective.append(event)
    counts = {kind: sum(event["decision"] == kind for event in effective) for kind in ("SAME_PROCESS", "DIFFERENT_PROCESS")}
    if not counts["SAME_PROCESS"] or not counts["DIFFERENT_PROCESS"]:
        raise ValueError("External affinity supervision requires at least one reviewed SAME and DIFFERENT decision")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable supervision: {output}")
    output.mkdir(parents=True)
    targets_path, mask_path = output / "affinity_targets_czyx.npy", output / "supervision_mask_czyx.npy"
    np.save(targets_path, targets, allow_pickle=False); np.save(mask_path, mask, allow_pickle=False)
    receipt = {
        "schema_version": 1, "id": f"MV-EXTERNAL-REVIEWED-AFFINITY-{workspace['crop_id'][-6:]}", "created_at": _now(),
        "status": "REVIEWED_DVID_AFFINITY_SUPERVISION", "review_state": "EXTERNAL_EXPERT_REVIEWED",
        "crop_id": workspace["crop_id"], "split": workspace["split"], "parent_region_id": workspace["parent_region_id"],
        "raw_sha256": workspace["raw"]["sha256"], "event_log": {"path": str(log.resolve()), "sha256": sha256_file(log), "events": len(events)},
        "counts": {**counts, "mask_pairs": int(mask.sum()), "ignored_pairs": int(mask.size - mask.sum())},
        "effective_decisions": effective, "excluded_placeholder_reviewer_events": excluded_placeholder_reviewer_events,
        "targets": {"path": str(targets_path.resolve()), "sha256": sha256_file(targets_path), "shape_czyx": list(targets.shape), "dtype": str(targets.dtype)},
        "mask": {"path": str(mask_path.resolve()), "sha256": sha256_file(mask_path), "shape_czyx": list(mask.shape), "dtype": str(mask.dtype)},
        "prohibited_promotions": ["MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
    }
    write_json_atomic(output / "receipt.json", receipt)
    return receipt
