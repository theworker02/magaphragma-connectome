"""Convert append-only human boundary decisions into masked affinity targets."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file, write_json_atomic


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate_key(candidate: dict[str, Any]) -> str:
    """Stable identity across differently ranked queues.

    Queue display IDs are ordinal within a particular ranking run, so they are
    not suitable as a scientific decision identity when a second sampler is
    reviewed.  The reviewed interface is its segment pair and local point.
    """
    required = ("supervoxel_a", "supervoxel_b", "local_point_zyx")
    if any(key not in candidate for key in required):
        raise ValueError("Review decision lacks a complete candidate interface")
    return f"{min(int(candidate['supervoxel_a']), int(candidate['supervoxel_b']))}:{max(int(candidate['supervoxel_a']), int(candidate['supervoxel_b']))}:{','.join(map(str, candidate['local_point_zyx']))}"


def _nearest_face(labels: np.ndarray, a_id: int, b_id: int, preferred: list[int]) -> tuple[int, tuple[int, int, int]] | None:
    """Return channel (Z/Y/X) and left voxel for the nearest A/B face."""
    result: list[tuple[int, np.ndarray]] = []
    for axis in range(3):
        left = [slice(None)] * 3; right = [slice(None)] * 3; left[axis] = slice(0, -1); right[axis] = slice(1, None)
        lhs, rhs = labels[tuple(left)], labels[tuple(right)]
        points = np.argwhere(((lhs == a_id) & (rhs == b_id)) | ((lhs == b_id) & (rhs == a_id)))
        if len(points): result.append((axis, points))
    if not result: return None
    reference = np.asarray(preferred)
    axis, points = min(result, key=lambda item: int(((item[1] - reference) ** 2).sum(axis=1).min()))
    point = points[np.argmin(((points - reference) ** 2).sum(axis=1))]
    return axis, tuple(int(value) for value in point)


def materialize_reviewed_affinity(workspace_path: Path, output: Path) -> dict[str, Any]:
    """Create immutable masked targets from a real review log.

    Repeated identical clicks are retained in the raw log but collapse to one
    effective decision. Conflicting clicks and UNCERTAIN/BAD_QUESTION events
    remain excluded from the mask. The result is supervision, never a neuron
    or instance reconstruction.
    """
    workspace = json.loads(workspace_path.read_text(encoding="utf-8"))
    if workspace.get("parent_region_id") == "MV-GTVOL-000004":
        raise ValueError("Regression-only cube cannot become reviewed supervision")
    if workspace.get("status") != "MACHINE_SUPERVOXEL_GRAPH_REVIEW_REQUIRED":
        raise ValueError("Expected a machine supervoxel review workspace")
    log_path = Path(workspace["edit_log"]["path"])
    if not log_path.exists(): raise ValueError("No reviewer event log exists")
    events = _events(log_path)
    decisions = [event for event in events if event.get("kind") in {"SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"}]
    if not decisions: raise ValueError("No boundary decisions available")
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for event in decisions:
        candidate = event.get("payload", {}).get("candidate", {})
        by_candidate.setdefault(_candidate_key(candidate), []).append(event)
    labels = np.load(workspace["machine_supervoxels"]["path"], mmap_mode="r", allow_pickle=False)
    targets = np.zeros((3,) + labels.shape, dtype=np.uint8)
    mask = np.zeros((3,) + labels.shape, dtype=np.uint8)
    effective: list[dict[str, Any]] = []; conflicts: list[str] = []; excluded: list[str] = []
    for candidate_key, group in by_candidate.items():
        kinds = {event["kind"] for event in group}
        if len(kinds) != 1 or kinds & {"UNCERTAIN", "BAD_QUESTION"}:
            (conflicts if len(kinds) != 1 else excluded).append(candidate_key); continue
        event = group[-1]; candidate = event["payload"]["candidate"]
        face = _nearest_face(labels, int(candidate["supervoxel_a"]), int(candidate["supervoxel_b"]), candidate["local_point_zyx"])
        if face is None:
            excluded.append(candidate_key); continue
        channel, point = face; target = 1 if event["kind"] == "SAME_PROCESS" else 0
        targets[(channel,) + point] = target; mask[(channel,) + point] = 1
        effective.append({"candidate_id": candidate.get("id"), "candidate_interface_key": candidate_key, "kind": event["kind"], "channel_zyx": channel, "pair_left_zyx": list(point), "reviewer": event["reviewer"], "event_id": event["id"]})
    counts = {kind: sum(item["kind"] == kind for item in effective) for kind in ("SAME_PROCESS", "DIFFERENT_PROCESS")}
    if not counts["SAME_PROCESS"] or not counts["DIFFERENT_PROCESS"]:
        raise ValueError("Reviewed affinity supervision requires at least one effective SAME and DIFFERENT decision")
    if output.exists(): raise FileExistsError(f"Refusing to overwrite immutable reviewed supervision {output}")
    output.mkdir(parents=True)
    targets_path = output / "affinity_targets_czyx.npy"; mask_path = output / "supervision_mask_czyx.npy"
    np.save(targets_path, targets, allow_pickle=False); np.save(mask_path, mask, allow_pickle=False)
    result = {"schema_version": 1, "id": f"MV-REVIEWED-AFFINITY-{workspace['crop_id'][-6:]}", "created_at": _now(),
              "status": "REVIEWED_DVID_AFFINITY_SUPERVISION", "review_state": "HUMAN_REVIEWED",
              "crop_id": workspace["crop_id"], "parent_region_id": workspace["parent_region_id"], "split": "DVID_TARGET_TRAIN_OR_VALIDATION_FROM_CROP_MANIFEST",
              "raw_sha256": workspace["raw"]["sha256"], "supervoxels_sha256": workspace["machine_supervoxels"]["sha256"],
              "event_log": {"path": str(log_path), "sha256": sha256_file(log_path), "raw_events": len(events), "boundary_events": len(decisions)},
              "effective_decisions": effective, "duplicate_same_kind_events": len(decisions) - len(by_candidate), "conflicted_candidates": conflicts, "excluded_candidates": excluded,
              "counts": {**counts, "mask_pairs": int(mask.sum()), "ignored_pairs": int(mask.size - mask.sum())},
              "targets": {"path": str(targets_path.resolve()), "sha256": sha256_file(targets_path), "dtype": str(targets.dtype), "shape_czyx": list(targets.shape)},
              "mask": {"path": str(mask_path.resolve()), "sha256": sha256_file(mask_path), "dtype": str(mask.dtype), "shape_czyx": list(mask.shape)},
              "prohibited_promotions": ["MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"], "note": "Only explicitly reviewed face contacts are supervised; all other pairs are IGNORE."}
    write_json_atomic(output / "receipt.json", result)
    return result
