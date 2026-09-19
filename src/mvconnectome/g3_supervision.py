"""Freeze G3 interface-reviewed DVID affinity supervision.

The reviewer decision is made once per raw-EM interface group.  Direct
adjacent affinity pairs give that group local support, but the loss weight of
all of its member pairs sums to one so correlated observations cannot dominate
training.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file, write_json_atomic


_DECISIONS = {"SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"}
_PLACEHOLDERS = {"expert-identifier", "reviewer", "unknown", "tbd", "to-be-assigned"}
# Only an identified human reviewer can contribute to this dataset.  These
# placeholder identities remain audit records, never training evidence.


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        max(left["bounds_xyz"][axis][0], right["bounds_xyz"][axis][0])
        < min(left["bounds_xyz"][axis][1], right["bounds_xyz"][axis][1])
        for axis in ("x", "y", "z")
    )


def _interior(left: tuple[int, int, int], channel: int, shape: tuple[int, int, int]) -> bool:
    right = list(left)
    right[channel] += 1
    return all(0 < value < limit - 1 for value, limit in zip(left, shape, strict=True)) and all(
        0 < value < limit - 1 for value, limit in zip(right, shape, strict=True)
    )


def _validate_regions(cohort: dict[str, Any]) -> list[dict[str, Any]]:
    regions = cohort.get("regions", [])
    train = [item for item in regions if item.get("role") == "G3_TARGET_TRAIN"]
    validation = [item for item in regions if item.get("role") == "G3_TARGET_VALIDATION"]
    if len(regions) != 9 or len(train) != 6 or len(validation) != 3:
        raise ValueError("G3 requires exactly six TRAIN and three independent VALIDATION regions")
    if len({item.get("id") for item in regions}) != len(regions):
        raise ValueError("G3 region IDs are not unique")
    # 000004 is a one-time protected regression volume, not a development crop.
    if any("000004" in json.dumps(item, sort_keys=True) for item in regions):
        raise ValueError("Protected MV-GTVOL-000004 cannot enter G3 supervision")
    for index, left in enumerate(regions):
        for right in regions[index + 1:]:
            if _overlap(left, right):
                raise ValueError(f"G3 spatial overlap: {left['id']} / {right['id']}")
    return regions


def _load_region(
    region: dict[str, Any], *, workspace_path: Path, queue_path: Path, output_root: Path
) -> dict[str, Any]:
    # The workspace freezes raw-crop provenance; the queue freezes exactly
    # which local interfaces the reviewer was asked to judge.
    workspace, queue = _json(workspace_path), _json(queue_path)
    crop_id, raw_hash = region["id"], region["raw_sha256"]
    if workspace.get("crop_id") != crop_id or workspace.get("split") != region["role"]:
        raise ValueError(f"{crop_id}: workspace crop/split mismatch")
    if workspace.get("parent_region_id") == "MV-GTVOL-000004" or workspace.get("raw", {}).get("sha256") != raw_hash:
        raise ValueError(f"{crop_id}: protected volume or raw hash mismatch")
    raw_path = Path(workspace["raw"]["path"])
    if not raw_path.is_file() or sha256_file(raw_path) != raw_hash:
        raise ValueError(f"{crop_id}: immutable raw source changed")
    shape = tuple(int(value) for value in workspace["raw"]["shape_zyx"])
    if shape != tuple(int(value) for value in region["shape_zyx"]):
        raise ValueError(f"{crop_id}: raw geometry mismatch")
    if queue.get("status") != "EXPERT_INTERFACE_REVIEW_REQUIRED" or queue.get("crop_id") != crop_id:
        raise ValueError(f"{crop_id}: invalid frozen interface queue")
    if queue.get("workspace_id") != workspace.get("id") or queue.get("raw_sha256") != raw_hash:
        raise ValueError(f"{crop_id}: queue/workspace provenance mismatch")
    questions = queue.get("questions", [])
    question_index = {item.get("id"): item for item in questions}
    if not questions or len(question_index) != len(questions):
        raise ValueError(f"{crop_id}: invalid or duplicate frozen questions")
    log_path = Path(workspace["event_log"]["path"])
    if not log_path.is_file():
        raise ValueError(f"{crop_id}: missing append-only review log")
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen_questions: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        # An event must point back to one—and only one—pre-frozen question.
        # This prevents post-hoc pair insertion after seeing training results.
        reference, decision = event.get("question_reference"), event.get("decision")
        if reference not in question_index or reference in seen_questions:
            raise ValueError(f"{crop_id}: missing, duplicate, or unknown question reference")
        seen_questions.add(reference)
        if decision not in _DECISIONS or str(event.get("reviewer", "")).strip().casefold() in _PLACEHOLDERS:
            raise ValueError(f"{crop_id}: invalid decision or placeholder reviewer")
        question = question_index[reference]
        if event.get("crop_id") != crop_id or event.get("split") != region["role"] or event.get("raw_sha256") != raw_hash:
            raise ValueError(f"{crop_id}: event provenance mismatch")
        if event.get("interface_id") != question.get("interface_id"):
            raise ValueError(f"{crop_id}: interface provenance mismatch")
        channel, left = int(event.get("channel_zyx")), tuple(int(value) for value in event.get("pair_left_zyx", []))
        if channel != int(question.get("channel_zyx")) or list(left) != question.get("pair_left_zyx"):
            raise ValueError(f"{crop_id}: event affinity geometry differs from frozen question")
        right = list(left); right[channel] += 1
        if channel not in (0, 1, 2) or len(left) != 3 or list(event.get("pair_right_zyx", [])) != right:
            raise ValueError(f"{crop_id}: invalid direct affinity pair")
        if any(value < 0 or value >= limit for value, limit in zip(left, shape, strict=True)) or right[channel] >= shape[channel]:
            raise ValueError(f"{crop_id}: out-of-bounds direct affinity pair")
        grouped.setdefault(str(event["interface_id"]), []).append(event)
    if set(question_index) != seen_questions:
        raise ValueError(f"{crop_id}: review log is incomplete")

    targets = np.zeros((3,) + shape, dtype=np.uint8)
    mask = np.zeros((3,) + shape, dtype=np.uint8)
    weights = np.zeros((3,) + shape, dtype=np.float32)
    interface_records: list[dict[str, Any]] = []
    pair_keys: set[tuple[int, tuple[int, int, int]]] = set()
    eligible_pairs: list[dict[str, Any]] = []
    for interface_id, members in sorted(grouped.items()):
        # Interface, rather than direct voxel pair, is the biological evidence
        # unit.  Any mixed decision invalidates the entire interface; we never
        # cherry-pick the convenient member pairs.
        decisions = {str(member["decision"]) for member in members}
        usable = len(decisions) == 1 and next(iter(decisions)) in {"SAME_PROCESS", "DIFFERENT_PROCESS"}
        interior_members = [member for member in members if _interior(tuple(member["pair_left_zyx"]), int(member["channel_zyx"]), shape)]
        admitted_members = interior_members if usable else []
        decision = next(iter(decisions)) if len(decisions) == 1 else "CONTRADICTORY"
        interface_records.append({
            "interface_id": interface_id,
            "decision": decision,
            "member_questions": [member["question_reference"] for member in members],
            "member_pair_count": len(members),
            "interior_pair_count": len(interior_members),
            "eligible_pair_count": len(admitted_members),
            "eligible": bool(admitted_members),
            "exclusion_reason": None if admitted_members else ("MIXED_OR_NON_SUPERVISED_INTERFACE_DECISION" if not usable else "NO_INTERIOR_PAIR"),
        })
        if not admitted_members:
            continue
        # Correlated pairs around one reviewed membrane share a total weight of
        # one.  A denser local sample therefore cannot dominate optimisation.
        weight = 1.0 / len(admitted_members)
        for member in admitted_members:
            channel, left = int(member["channel_zyx"]), tuple(int(value) for value in member["pair_left_zyx"])
            key = (channel, left)
            if key in pair_keys:
                raise ValueError(f"{crop_id}: duplicate canonical affinity pair")
            pair_keys.add(key)
            targets[(channel,) + left] = 1 if decision == "SAME_PROCESS" else 0
            mask[(channel,) + left] = 1
            weights[(channel,) + left] = weight
            eligible_pairs.append({
                "interface_id": interface_id,
                "question_reference": member["question_reference"],
                "reviewer": member["reviewer"],
                "decision": decision,
                "channel_zyx": channel,
                "pair_left_zyx": list(left),
                "pair_right_zyx": member["pair_right_zyx"],
                "loss_weight": weight,
                "raw_sha256": raw_hash,
            })
    counts = {decision: sum(item["decision"] == decision for item in eligible_pairs) for decision in ("SAME_PROCESS", "DIFFERENT_PROCESS")}
    interface_counts = {decision: sum(item["eligible"] and item["decision"] == decision for item in interface_records) for decision in ("SAME_PROCESS", "DIFFERENT_PROCESS")}
    # Both classes are mandatory after all edge and ambiguity exclusions.
    if not counts["SAME_PROCESS"] or not counts["DIFFERENT_PROCESS"]:
        raise ValueError(f"{crop_id}: no eligible interior SAME/DIFFERENT evidence after interface exclusions")
    if not np.isclose(float(weights.sum()), float(sum(interface_counts.values()))):
        raise ValueError(f"{crop_id}: interface loss normalization failed")
    region_dir = output_root / crop_id
    if region_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable G3 supervision: {region_dir}")
    region_dir.mkdir(parents=True)
    target_path, mask_path, weight_path = region_dir / "affinity_targets_czyx.npy", region_dir / "supervision_mask_czyx.npy", region_dir / "interface_weights_czyx.npy"
    np.save(target_path, targets, allow_pickle=False)
    np.save(mask_path, mask, allow_pickle=False)
    np.save(weight_path, weights, allow_pickle=False)
    materializer = Path(__file__).resolve()
    receipt = {
        "schema_version": 1, "id": f"MV-G3-REVIEWED-INTERFACE-AFFINITY-{crop_id[-6:]}", "created_at": _now(),
        "status": "REVIEWED_DVID_INTERFACE_AFFINITY_SUPERVISION", "crop_id": crop_id, "split": region["role"],
        "raw_sha256": raw_hash, "workspace": {"path": str(workspace_path.resolve()), "sha256": sha256_file(workspace_path)},
        "queue": {"path": str(queue_path.resolve()), "sha256": sha256_file(queue_path)},
        "review_log": {"path": str(log_path.resolve()), "sha256": sha256_file(log_path), "events": len(events)},
        "materializer": {"path": str(materializer), "sha256": sha256_file(materializer), "contract": "INTERFACE_LEVEL_FAIL_CLOSED_V1"},
        "counts": {"eligible_pairs": counts, "eligible_interfaces": interface_counts, "ignored_pairs": int(mask.size - mask.sum())},
        "interfaces": interface_records, "eligible_pairs": eligible_pairs,
        "targets": {"path": str(target_path.resolve()), "sha256": sha256_file(target_path), "shape_czyx": list(targets.shape), "dtype": str(targets.dtype)},
        "mask": {"path": str(mask_path.resolve()), "sha256": sha256_file(mask_path), "shape_czyx": list(mask.shape), "dtype": str(mask.dtype)},
        "weights": {"path": str(weight_path.resolve()), "sha256": sha256_file(weight_path), "shape_czyx": list(weights.shape), "dtype": str(weights.dtype), "normalization": "SUM_OF_WEIGHTS_PER_ELIGIBLE_INTERFACE_EQUALS_ONE"},
        "prohibited_promotions": ["MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
    }
    write_json_atomic(region_dir / "receipt.json", receipt)
    return receipt


def freeze_g3_supervision(*, cohort_path: Path, inputs_path: Path, output_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Materialize a frozen G3 dataset from reviewed interface queues."""
    # Immutable output is intentional: a later model result must never rewrite
    # labels, masks, weights, or the cohort that produced it.
    if manifest_path.exists() or output_root.exists():
        raise FileExistsError("Refusing to overwrite an immutable G3 supervision dataset")
    cohort, inputs = _json(cohort_path), _json(inputs_path)
    regions = _validate_regions(cohort)
    bindings = inputs.get("regions", {})
    if set(bindings) != {region["id"] for region in regions}:
        raise ValueError("G3 workspace/queue bindings do not exactly match the frozen cohort")
    receipts = []
    for region in regions:
        binding = bindings[region["id"]]
        receipts.append(_load_region(region, workspace_path=Path(binding["workspace"]), queue_path=Path(binding["queue"]), output_root=output_root))
    result = {
        "schema_version": 1, "id": manifest_path.stem, "created_at": _now(), "status": "FROZEN_READY_FOR_G3_LORO",
        "cohort": {"path": str(cohort_path.resolve()), "sha256": sha256_file(cohort_path)},
        "input_bindings": {"path": str(inputs_path.resolve()), "sha256": sha256_file(inputs_path)},
        "regions": [{
            "crop_id": receipt["crop_id"], "split": receipt["split"], "raw_sha256": receipt["raw_sha256"],
            "raw_path": _json(Path(receipt["workspace"]["path"]))["raw"]["path"],
            "receipt": str((output_root / receipt["crop_id"] / "receipt.json").resolve()),
            "receipt_sha256": sha256_file(output_root / receipt["crop_id"] / "receipt.json"),
            "eligible_pairs": receipt["counts"]["eligible_pairs"], "eligible_interfaces": receipt["counts"]["eligible_interfaces"],
            "targets": receipt["targets"], "mask": receipt["mask"], "weights": receipt["weights"],
        } for receipt in receipts],
        "materializer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve()), "contract": "INTERFACE_LEVEL_FAIL_CLOSED_V1"},
        "loro_folds": cohort["loro_folds"], "protected_regression": {"volume": "MV-GTVOL-000004", "included": False},
        "loss_contract": "Only supervised direct pairs are used. Each eligible reviewed interface has total loss weight one; validation is evaluation-only.",
        "biological_promotions": {"MV-FRAG": 0, "MV-N": 0, "MV-SYN": 0, "MV-CONN": 0},
    }
    write_json_atomic(manifest_path, result)
    return result
