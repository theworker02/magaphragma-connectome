"""Raw-EM-only membrane-crossing question generation for human review."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file, write_json_atomic


def _now() -> str: return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def materialize_raw_membrane_pilot(amendment: Path, parents: Path, output: Path) -> dict[str, Any]:
    """Copy the frozen crop once and write its immutable source receipt."""
    plan = json.loads(amendment.read_text(encoding="utf-8")); crop = plan["crop"]
    if crop["parent_region_id"] == "MV-GTVOL-000004": raise ValueError("Regression-only volume is prohibited")
    parent = next(item for item in json.loads(parents.read_text(encoding="utf-8"))["regions"] if item["id"] == crop["parent_region_id"])
    raw = np.load(parent["raw_path"], mmap_mode="r", allow_pickle=False); z, y, x = crop["origin_zyx"]; dz, dy, dx = crop["shape_zyx"]
    if z + dz > raw.shape[0] or y + dy > raw.shape[1] or x + dx > raw.shape[2]: raise ValueError("Frozen crop exceeds parent")
    target = output / "raw_zyx.npy"
    if target.exists(): raise FileExistsError("Refusing to overwrite immutable pilot raw crop")
    output.mkdir(parents=True, exist_ok=True); np.save(target, np.asarray(raw[z:z+dz, y:y+dy, x:x+dx]).copy(), allow_pickle=False)
    absolute_xyz = [parent["bounds_xyz"]["x"][0] + x, parent["bounds_xyz"]["y"][0] + y, parent["bounds_xyz"]["z"][0] + z]
    receipt = {"id": "MV-DVID-ANN-000005-MEMBRANE-PILOT-001", "status": "RAW_EM_MEMBRANE_PILOT_RAW_IMMUTABLE", "created_at": _now(),
               "parent_region_id": parent["id"], "parent_raw_sha256": parent["raw_sha256"], "crop_origin_zyx": crop["origin_zyx"], "crop_shape_zyx": crop["shape_zyx"],
               "absolute_origin_xyz": absolute_xyz, "voxel_size_nm_xyz": [8.0, 8.0, 8.0], "coordinate_frame": parent["coordinate_frame"],
               "raw_path": str(target.resolve()), "raw_sha256": sha256_file(target), "dtype": str(raw.dtype), "protected_overlap": "NONE"}
    write_json_atomic(output / "raw-receipt.json", receipt); return receipt


def generate_raw_membrane_questions(raw_receipt: Path, output: Path, maximum: int = 20, minimum_separation: int = 12) -> dict[str, Any]:
    """Generate spatially-suppressed candidate faces solely from raw 3-D EM.

    A candidate is a high, locally orientation-consistent intensity gradient.
    Its A/B pair is always adjacent along one Z/Y/X axis, so no diagonal label
    is ever silently projected into affinity supervision.
    """
    receipt = json.loads(raw_receipt.read_text(encoding="utf-8")); raw = np.load(receipt["raw_path"], allow_pickle=False).astype(np.float32)
    if raw.ndim != 3: raise ValueError("Expected raw ZYX volume")
    gradient = np.stack(np.gradient(raw), axis=0); magnitude = np.sqrt((gradient ** 2).sum(axis=0))
    axes = np.argmax(np.abs(gradient), axis=0); border = 5; valid = np.zeros(raw.shape, bool); valid[border:-border, border:-border, border:-border] = True
    # 3-D orientation consistency: compare local gradients to their 6-neighbor mean.
    normalized = gradient / np.maximum(magnitude, 1e-6)[None]
    neighbor = sum(np.roll(normalized, shift, axis=axis + 1) for axis in range(3) for shift in (-1, 1)) / 6.0
    consistency = np.abs((normalized * neighbor).sum(axis=0)); threshold = np.percentile(magnitude[valid], 99.5)
    points = np.argwhere(valid & (magnitude >= threshold) & (consistency >= 0.55))
    order = points[np.argsort(magnitude[tuple(points.T)])[::-1]]; selected: list[dict[str, Any]] = []
    for point in order:
        axis = int(axes[tuple(point)]); other = point.copy(); other[axis] += 1
        if other[axis] >= raw.shape[axis] - border: continue
        if any(np.linalg.norm(point - np.asarray(item["center_zyx"])) < minimum_separation for item in selected): continue
        normal = np.zeros(3, dtype=int); normal[axis] = 1
        selected.append({"id": f"MV-MEM-Q-{len(selected)+1:03d}", "status": "MEMBRANE_CANDIDATE", "center_zyx": [int(v) for v in point],
                         "normal_zyx": normal.tolist(), "a_zyx": [int(v) for v in point], "b_zyx": [int(v) for v in other], "affinity_channel_zyx": axis,
                         "distance_voxels": 1, "distance_nm": 8.0, "raw_feature_score": float(magnitude[tuple(point)]),
                         "orientation_confidence": float(consistency[tuple(point)]), "neighborhood_quality": "RAW_3D_GRADIENT_ORIENTATION_CANDIDATE_NOT_BIOLOGICAL_LABEL"})
        if len(selected) == maximum: break
    if len(selected) != maximum: raise ValueError(f"Only {len(selected)} raw-EM candidates met frozen conservative criteria")
    if output.exists(): raise FileExistsError("Refusing to overwrite frozen membrane pilot")
    coverage = [item["affinity_channel_zyx"] for item in selected]
    result = {"id": receipt["id"], "status": "MEMBRANE_CANDIDATE_REVIEW_REQUIRED", "created_at": _now(), "raw_receipt_sha256": sha256_file(raw_receipt), "raw_sha256": receipt["raw_sha256"],
              "algorithm": {"name": "RAW_3D_GRADIENT_ORIENTATION_V1", "inputs": ["raw_em_only"], "prohibited_inputs": ["segneuron", "supervoxels", "prior_labels"], "minimum_separation_voxels": minimum_separation},
              "questions": selected, "coverage": {"count": len(selected), "axis_zyx": {str(axis): coverage.count(axis) for axis in range(3)}},
              "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"]}
    write_json_atomic(output, result); return result


def generate_raw_continuity_control_questions(raw_receipt: Path, output: Path, maximum: int = 6, minimum_separation: int = 32) -> dict[str, Any]:
    """Propose spatially separated low-gradient raw-EM pairs for review.

    These are *continuity candidates*, not SAME labels.  They deliberately use
    raw intensity geometry only and are reviewed with the same choices as
    boundary candidates.  The distinct status prevents a later consumer from
    mistaking a low-gradient proposal for a biological continuity decision.
    """
    receipt = json.loads(raw_receipt.read_text(encoding="utf-8"))
    raw = np.load(receipt["raw_path"], allow_pickle=False).astype(np.float32)
    if raw.ndim != 3:
        raise ValueError("Expected raw ZYX volume")
    gradient = np.stack(np.gradient(raw), axis=0)
    magnitude = np.sqrt((gradient ** 2).sum(axis=0))
    border = 8
    valid = np.zeros(raw.shape, bool)
    valid[border:-border, border:-border, border:-border] = True
    magnitude_valid = magnitude[valid]
    low, high = np.percentile(magnitude_valid, (20.0, 55.0))
    raw_low, raw_high = np.percentile(raw[valid], (5.0, 95.0))
    # Exclude both blank/background extremes and high-gradient interfaces.
    candidates = np.argwhere(valid & (magnitude >= low) & (magnitude <= high) & (raw >= raw_low) & (raw <= raw_high))
    # Prefer the most interior candidate pairs (smallest local gradient), but
    # retain deterministic spatial coverage through the separation rule.
    order = candidates[np.argsort(magnitude[tuple(candidates.T)])]
    selected: list[dict[str, Any]] = []
    for point in order:
        axis = int(np.argmin(np.abs(gradient[(slice(None),) + tuple(point)])))
        other = point.copy(); other[axis] += 1
        if other[axis] >= raw.shape[axis] - border:
            continue
        if any(np.linalg.norm(point - np.asarray(item["center_zyx"])) < minimum_separation for item in selected):
            continue
        normal = np.zeros(3, dtype=int); normal[axis] = 1
        selected.append({
            "id": f"MV-CONTROL-Q-{len(selected) + 1:03d}",
            "status": "CONTINUITY_CANDIDATE",
            "center_zyx": [int(value) for value in point],
            "normal_zyx": normal.tolist(),
            "a_zyx": [int(value) for value in point],
            "b_zyx": [int(value) for value in other],
            "affinity_channel_zyx": axis,
            "distance_voxels": 1,
            "distance_nm": 8.0,
            "raw_feature_score": float(magnitude[tuple(point)]),
            "orientation_confidence": None,
            "neighborhood_quality": "RAW_LOW_GRADIENT_CONTINUITY_CANDIDATE_NOT_BIOLOGICAL_LABEL",
        })
        if len(selected) == maximum:
            break
    if len(selected) != maximum:
        raise ValueError(f"Only {len(selected)} raw-EM continuity candidates met frozen criteria")
    if output.exists():
        raise FileExistsError("Refusing to overwrite frozen continuity controls")
    coverage = [item["affinity_channel_zyx"] for item in selected]
    result = {
        "id": receipt["id"], "status": "CONTINUITY_CANDIDATE_REVIEW_REQUIRED", "created_at": _now(),
        "raw_receipt_sha256": sha256_file(raw_receipt), "raw_sha256": receipt["raw_sha256"],
        "algorithm": {"name": "RAW_LOW_GRADIENT_CONTINUITY_V1", "inputs": ["raw_em_only"],
                      "prohibited_inputs": ["segneuron", "supervoxels", "prior_labels"],
                      "minimum_separation_voxels": minimum_separation,
                      "raw_gradient_percentile_band": [20.0, 55.0]},
        "questions": selected,
        "coverage": {"count": len(selected), "axis_zyx": {str(axis): coverage.count(axis) for axis in range(3)}},
        "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
    }
    write_json_atomic(output, result)
    return result


def create_membrane_review_workspace(*, raw_receipt: Path, questions: Path, output: Path) -> dict[str, Any]:
    """Freeze an append-only, raw-EM membrane-question review workspace.

    This workspace deliberately does not contain a segmentation or a label
    volume.  It is only a ledger for a human's local SAME/DIFFERENT judgement
    of frozen adjacent voxel pairs.
    """
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite review workspace {output}")
    raw = json.loads(raw_receipt.read_text(encoding="utf-8"))
    queue = json.loads(questions.read_text(encoding="utf-8"))
    if raw["id"] != queue["id"] or raw["raw_sha256"] != queue["raw_sha256"]:
        raise ValueError("Raw receipt and membrane questions do not describe the same immutable crop")
    if raw["parent_region_id"] == "MV-GTVOL-000004":
        raise ValueError("Regression-only volume cannot enter membrane review")
    if queue.get("status") not in {"MEMBRANE_CANDIDATE_REVIEW_REQUIRED", "CONTINUITY_CANDIDATE_REVIEW_REQUIRED"} or not queue.get("questions"):
        raise ValueError("Questions are not a frozen raw-EM review queue")
    value = {
        "schema_version": 1,
        "id": f"MV-MEMBRANE-REVIEW-{raw['id'][-3:]}",
        "created_at": _now(),
        "status": "RAW_EM_MEMBRANE_REVIEW_REQUIRED",
        "review_state": "UNREVIEWED",
        "crop_id": raw["id"],
        "parent_region_id": raw["parent_region_id"],
        "coordinate_frame": raw["coordinate_frame"],
        "raw": {"path": raw["raw_path"], "sha256": raw["raw_sha256"], "shape_zyx": raw["crop_shape_zyx"]},
        "questions": {"path": str(questions.resolve()), "sha256": sha256_file(questions), "count": len(queue["questions"])},
        "raw_receipt": {"path": str(raw_receipt.resolve()), "sha256": sha256_file(raw_receipt)},
        "decision_log": {"path": str((output.parent / f"{output.stem}.events.jsonl").resolve()), "append_only": True},
        "allowed_decisions": ["SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"],
        "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
    }
    write_json_atomic(output, value)
    return value


def append_membrane_review_event(workspace_path: Path, *, reviewer: str, question_id: str, decision: str) -> dict[str, Any]:
    """Append one human decision for a frozen raw-EM crossing question."""
    workspace = json.loads(workspace_path.read_text(encoding="utf-8"))
    reviewer = reviewer.strip()
    if workspace.get("status") != "RAW_EM_MEMBRANE_REVIEW_REQUIRED":
        raise ValueError("Workspace is not eligible for raw-EM membrane review")
    if not reviewer or decision not in set(workspace["allowed_decisions"]):
        raise ValueError("A reviewer and supported membrane decision are required")
    queue = json.loads(Path(workspace["questions"]["path"]).read_text(encoding="utf-8"))
    question = next((item for item in queue["questions"] if item["id"] == question_id), None)
    if question is None:
        raise ValueError("Decision references a question outside this frozen workspace")
    log = Path(workspace["decision_log"]["path"])
    if log.exists():
        prior = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(item.get("question_id") == question_id for item in prior):
            raise ValueError("A frozen membrane question already has an append-only decision")
    event = {
        "id": f"MV-MEMBRANE-REVIEW-EVENT-{int(datetime.now(UTC).timestamp() * 1_000_000)}",
        "timestamp": _now(), "workspace_id": workspace["id"], "question_id": question_id,
        "crop_id": workspace["crop_id"], "center_zyx": question["center_zyx"], "a_zyx": question["a_zyx"],
        "b_zyx": question["b_zyx"], "normal_zyx": question["normal_zyx"],
        "affinity_channel_zyx": question["affinity_channel_zyx"],
        "raw_features": {key: question[key] for key in ("raw_feature_score", "orientation_confidence", "neighborhood_quality")},
        "algorithm": queue.get("algorithm", {"name": "UNKNOWN_FROZEN_QUEUE_ALGORITHM"}), "raw_sha256": workspace["raw"]["sha256"],
        "reviewer": reviewer, "decision": decision,
        "effect": "REVIEWED_LOCAL_AFFINITY_EVIDENCE_NOT_A_BIOLOGICAL_PROMOTION",
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def materialize_reviewed_membrane_affinity(*, workspace_paths: list[Path], role: str, output: Path) -> dict[str, Any]:
    """Convert reviewed raw-EM pair decisions into immutable CZYX supervision.

    This accepts a boundary queue plus continuity controls for *one* immutable
    raw crop.  It fails closed unless they agree on crop identity, raw hash,
    geometry, and contain reviewed SAME and DIFFERENT relationships.
    """
    if not workspace_paths:
        raise ValueError("At least one review workspace is required")
    workspaces = [json.loads(path.read_text(encoding="utf-8")) for path in workspace_paths]
    first = workspaces[0]
    if any(workspace.get("status") != "RAW_EM_MEMBRANE_REVIEW_REQUIRED" for workspace in workspaces):
        raise ValueError("Every input must be a raw-EM review workspace")
    for key in ("crop_id", "parent_region_id", "coordinate_frame"):
        if any(workspace.get(key) != first.get(key) for workspace in workspaces):
            raise ValueError(f"Workspaces disagree on {key}")
    raw_metadata = first["raw"]
    if any(workspace["raw"].get("sha256") != raw_metadata.get("sha256") or workspace["raw"].get("shape_zyx") != raw_metadata.get("shape_zyx") for workspace in workspaces):
        raise ValueError("Workspaces do not reference the same immutable raw crop")
    raw_path = Path(raw_metadata["path"])
    if not raw_path.exists() or sha256_file(raw_path) != raw_metadata["sha256"]:
        raise ValueError("Raw crop is missing or its hash changed after review")
    # Use an ordinary read here.  A memmap keeps the source file handle open on
    # Windows when this function deliberately rejects malformed evidence.
    raw = np.load(raw_path, allow_pickle=False)
    if tuple(raw.shape) != tuple(raw_metadata["shape_zyx"]):
        raise ValueError("Raw crop shape no longer matches its workspace")
    targets = np.zeros((3,) + tuple(raw.shape), dtype=np.uint8)
    mask = np.zeros((3,) + tuple(raw.shape), dtype=np.uint8)
    decisions: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    # Canonical undirected endpoint keys make an accidental A→B / B→A copy a
    # hard failure.  Although the target tensor uses its verified positive-axis
    # convention, review evidence itself is semantically an undirected local
    # relationship and must not be counted twice.
    canonical_pairs: dict[tuple[tuple[int, int, int], tuple[int, int, int]], str] = {}
    placeholder_reviewers = {"", "expert-identifier", "reviewer", "unknown", "tbd", "to-be-assigned"}
    for path, workspace in zip(workspace_paths, workspaces, strict=True):
        raw_receipt_path = Path(workspace["raw_receipt"]["path"])
        if not raw_receipt_path.exists() or sha256_file(raw_receipt_path) != workspace["raw_receipt"]["sha256"]:
            raise ValueError("Frozen raw receipt changed after workspace creation")
        raw_receipt = json.loads(raw_receipt_path.read_text(encoding="utf-8"))
        if raw_receipt.get("raw_sha256") != raw_metadata["sha256"] or raw_receipt.get("id") != first["crop_id"]:
            raise ValueError("Workspace raw receipt no longer describes its immutable crop")
        questions_path = Path(workspace["questions"]["path"])
        if not questions_path.exists() or sha256_file(questions_path) != workspace["questions"]["sha256"]:
            raise ValueError("Frozen question queue changed after workspace creation")
        questions = {item["id"]: item for item in json.loads(questions_path.read_text(encoding="utf-8"))["questions"]}
        log_path = Path(workspace["decision_log"]["path"])
        if not log_path.exists():
            raise ValueError("A review workspace has no append-only decision log")
        events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        logs.append({"workspace_path": str(path.resolve()), "workspace_sha256": sha256_file(path), "event_log_path": str(log_path.resolve()), "event_log_sha256": sha256_file(log_path), "events": len(events)})
        for event in events:
            if event.get("raw_sha256") != raw_metadata["sha256"] or event.get("question_id") not in questions:
                raise ValueError("Review event does not match its immutable raw source or frozen question queue")
            if event.get("decision") not in {"SAME_PROCESS", "DIFFERENT_PROCESS"}:
                continue
            if str(event.get("reviewer", "")).strip().lower() in placeholder_reviewers:
                raise ValueError("Reviewed supervision requires a non-placeholder reviewer identity")
            channel = int(event["affinity_channel_zyx"])
            left = tuple(int(value) for value in event["a_zyx"])
            supplied_right = tuple(int(value) for value in event.get("b_zyx", ()))
            if channel not in (0, 1, 2) or len(left) != 3:
                raise ValueError("Review event has invalid affinity geometry")
            right = list(left); right[channel] += 1
            if any(value < 0 or value >= limit for value, limit in zip(left, raw.shape, strict=True)) or right[channel] >= raw.shape[channel]:
                raise ValueError("Review event lies outside immutable raw crop")
            if supplied_right != tuple(right):
                raise ValueError("Review event does not encode the verified positive-axis adjacency")
            question = questions[event["question_id"]]
            if tuple(question.get("a_zyx", ())) != left or tuple(question.get("b_zyx", ())) != supplied_right:
                raise ValueError("Review event pair geometry differs from its frozen question")
            index = (channel,) + left
            target = 1 if event["decision"] == "SAME_PROCESS" else 0
            canonical = tuple(sorted((left, tuple(right))))
            if canonical in canonical_pairs:
                prior = canonical_pairs[canonical]
                if prior != event["decision"]:
                    raise ValueError("Contradictory append-only decisions target the same canonical affinity pair")
                raise ValueError("Duplicate or reversed review event targets the same canonical affinity pair")
            canonical_pairs[canonical] = event["decision"]
            if mask[index]:
                raise ValueError("Duplicate target tensor index in reviewed affinity supervision")
            targets[index] = target; mask[index] = 1; decisions.append(event)
    counts = {decision: sum(event["decision"] == decision for event in decisions) for decision in ("SAME_PROCESS", "DIFFERENT_PROCESS")}
    if not counts["SAME_PROCESS"] or not counts["DIFFERENT_PROCESS"]:
        raise ValueError("Reviewed affinity supervision requires both SAME_PROCESS and DIFFERENT_PROCESS evidence")
    if output.exists():
        raise FileExistsError("Refusing to overwrite immutable reviewed affinity supervision")
    output.mkdir(parents=True)
    targets_path, mask_path = output / "affinity_targets_czyx.npy", output / "supervision_mask_czyx.npy"
    np.save(targets_path, targets, allow_pickle=False); np.save(mask_path, mask, allow_pickle=False)
    receipt = {
        "id": f"MV-DVID-REVIEWED-AFFINITY-{first['crop_id'][-3:]}", "created_at": _now(),
        "status": "REVIEWED_DVID_AFFINITY_SUPERVISION", "role": role,
        "crop_id": first["crop_id"], "parent_region_id": first["parent_region_id"], "coordinate_frame": first["coordinate_frame"],
        "raw": {"path": str(raw_path.resolve()), "sha256": raw_metadata["sha256"], "shape_zyx": list(raw.shape), "dtype": str(raw.dtype)},
        "review_logs": logs, "counts": {**counts, "effective_pairs": int(mask.sum()), "ignored_pairs": int(mask.size - mask.sum())},
        "effective_decisions": decisions,
        "targets": {"path": str(targets_path.resolve()), "sha256": sha256_file(targets_path), "shape_czyx": list(targets.shape), "dtype": str(targets.dtype)},
        "mask": {"path": str(mask_path.resolve()), "sha256": sha256_file(mask_path), "shape_czyx": list(mask.shape), "dtype": str(mask.dtype)},
        "prohibited_promotions": ["MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
    }
    write_json_atomic(output / "receipt.json", receipt)
    return receipt


def verify_reviewed_dvid_pair_partitions(*, train_receipt_path: Path, validation_receipt_path: Path,
                                         partitions_path: Path, output: Path) -> dict[str, Any]:
    """Fail closed before target adaptation on reviewed TRAIN/VALIDATION tensors.

    This is intentionally a mechanical gate: it re-hashes all immutable
    inputs, validates each tensor/receipt relationship, rejects protected
    regression lineage, and proves the frozen XYZ regions do not overlap.
    """
    if output.exists():
        raise FileExistsError("Refusing to overwrite immutable pair-supervision gate receipt")
    train = json.loads(train_receipt_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_receipt_path.read_text(encoding="utf-8"))
    partitions = json.loads(partitions_path.read_text(encoding="utf-8"))
    expected_roles = ((train, "DVID_TARGET_TRAIN"), (validation, "DVID_TARGET_VALIDATION"))
    assignments = {item["crop_id"]: item for item in partitions.get("assignments", [])}
    checked: list[dict[str, Any]] = []
    for receipt, expected_role in expected_roles:
        if receipt.get("role") != expected_role:
            raise ValueError(f"Receipt role must be {expected_role}")
        if receipt.get("parent_region_id") == "MV-GTVOL-000004" or "000004" in receipt.get("crop_id", ""):
            raise ValueError("Regression-only volume cannot enter DVID target supervision")
        assignment = assignments.get(receipt.get("crop_id"))
        if assignment is None or assignment.get("role") != f"{expected_role}_CANDIDATE":
            raise ValueError("Reviewed receipt is absent from frozen partition assignment")
        if not receipt.get("counts", {}).get("SAME_PROCESS") or not receipt.get("counts", {}).get("DIFFERENT_PROCESS"):
            raise ValueError("Both reviewed SAME and DIFFERENT evidence are required per partition")
        raw = receipt["raw"]
        raw_path = Path(raw["path"])
        if not raw_path.exists() or sha256_file(raw_path) != raw["sha256"]:
            raise ValueError("Reviewed receipt raw source is missing or mutated")
        for item_name in ("targets", "mask"):
            item = receipt[item_name]; item_path = Path(item["path"])
            if not item_path.exists() or sha256_file(item_path) != item["sha256"]:
                raise ValueError(f"Reviewed {item_name} tensor is missing or mutated")
            array = np.load(item_path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != item["shape_czyx"] or tuple(array.shape) != (3, *raw["shape_zyx"]):
                raise ValueError(f"Reviewed {item_name} tensor geometry is invalid")
        for log in receipt.get("review_logs", []):
            for label in ("workspace", "event_log"):
                path = Path(log[f"{label}_path"])
                if not path.exists() or sha256_file(path) != log[f"{label}_sha256"]:
                    raise ValueError(f"Frozen review {label} was mutated after materialization")
        checked.append({"crop_id": receipt["crop_id"], "receipt_path": str((train_receipt_path if receipt is train else validation_receipt_path).resolve()),
                        "receipt_sha256": sha256_file(train_receipt_path if receipt is train else validation_receipt_path),
                        "bounds_xyz": assignment["bounds_xyz"], "counts": receipt["counts"]})

    def overlaps(left: dict[str, list[int]], right: dict[str, list[int]]) -> bool:
        return all(max(left[axis][0], right[axis][0]) < min(left[axis][1], right[axis][1]) for axis in ("x", "y", "z"))
    if overlaps(checked[0]["bounds_xyz"], checked[1]["bounds_xyz"]):
        raise ValueError("Frozen TRAIN and VALIDATION regions overlap")
    result = {
        "id": "MV-DVID-REVIEWED-PAIR-SUPERVISION-GATE-001", "created_at": _now(),
        "status": "TRAIN_AND_VALIDATION_EVIDENCE_GATE_PASS", "partitions_path": str(partitions_path.resolve()),
        "partitions_sha256": sha256_file(partitions_path), "train": checked[0], "validation": checked[1],
        "checks": {"same_and_different_per_partition": "PASS", "raw_hashes": "PASS", "tensor_hashes_and_geometry": "PASS",
                   "frozen_workspace_and_log_hashes": "PASS", "canonical_pair_uniqueness": "PASS",
                   "train_validation_xyz_overlap": "NONE", "regression_only_exclusion": "PASS"},
        "g1_target_adaptation": "READY", "prohibited_promotions": ["MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, result)
    return result
