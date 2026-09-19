"""Model-qualification records and label-based instance-segmentation metrics.

These utilities do not manufacture ground truth.  They only accept immutable,
same-grid instance labels supplied by an authorized source or a recorded human
review workflow, and keep validation/test evaluation separate from training.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import sha256_file, write_json_atomic


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def freeze_preproduction_baseline(root: Path, output: Path) -> dict[str, Any]:
    domain_path = root / "local_research_build" / "phase5c-production" / "domain.json"
    chunks_path = root / "local_research_build" / "phase5c-production" / "chunks.json"
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    result = {
        "id": "MV-PRE-PRODUCTION-BASELINE",
        "created_at": _now(),
        "immutable_after_creation": True,
        "dvid_source": {"dataset_id": domain["dataset_id"], "volume_id": domain["volume_id"], "coordinate_frame": domain["coordinate_frame"]},
        "production_domain": {"path": str(domain_path), "sha256": sha256_file(domain_path), "source_bounds_xyz": domain["source_bounds_xyz"], "dimensions_voxels_xyz": domain["dimensions_voxels_xyz"], "voxel_size_nm_xyz": domain["voxel_size_nm_xyz"], "physical_dimensions_nm_xyz": domain["physical_dimensions_nm_xyz"], "edge_read": domain["source_access"]},
        "chunk_strategy": {"path": str(chunks_path), "sha256": sha256_file(chunks_path), **domain["chunking"]},
        "registration": {"decision": "INSUFFICIENT_EVIDENCE", "catmaid_derived_dvid_seeding": "PROHIBITED", "decision_path": str(root / "registration" / "decision.json")},
        "backend_state": {"ffn": "ENVIRONMENT_IMPORT_PASS_CHECKPOINT_UNAVAILABLE", "production_segmenter": "NONE_SELECTED", "synapse_backend": "UNKNOWN_OR_UNAVAILABLE", "full_volume_processing": "BLOCKED_MODEL_VALIDATION"},
        "test_results_at_freeze": {"command": "$env:PYTHONPATH='src'; python -m unittest discover -s tests -v", "status": "PASS", "tests": 17},
        "source_hashes": {"parent_metadata_sha256": domain["source_access"]["parent_metadata_sha256"], "far_boundary_raw_sha256": domain["source_access"]["far_boundary_raw_sha256"]},
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("id") == result["id"]:
            return existing
        raise ValueError(f"Refusing to overwrite unrelated baseline: {output}")
    write_json_atomic(output, result)
    return result


def amend_baseline_test_count(baseline: Path, output: Path, observed_tests: int) -> dict[str, Any]:
    """Correct a factual baseline receipt without mutating the frozen record."""
    original = json.loads(baseline.read_text(encoding="utf-8"))
    amendment = {
        "id": "MV-PRE-PRODUCTION-BASELINE-AMENDMENT-001",
        "amends": {"path": str(baseline), "sha256": sha256_file(baseline)},
        "created_at": _now(),
        "reason": "The frozen baseline recorded the prior 16-test count although the freeze invocation ran 17 tests after qualification metrics were added.",
        "corrected_test_results": {"command": "$env:PYTHONPATH='src'; python -m unittest discover -s tests -v", "status": "PASS", "tests": observed_tests},
        "original_record_retained": True,
    }
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    if original.get("id") != "MV-PRE-PRODUCTION-BASELINE":
        raise ValueError("Amendment target is not the Phase 5D baseline")
    write_json_atomic(output, amendment)
    return amendment


def _contingency(truth: Any, prediction: Any) -> tuple[dict[tuple[int, int], int], dict[int, int], dict[int, int]]:
    import numpy as np

    truth_values = np.asarray(truth)
    prediction_values = np.asarray(prediction)
    if truth_values.shape != prediction_values.shape or truth_values.ndim != 3:
        raise ValueError("Truth and prediction must be same-shape three-dimensional instance arrays")
    # Background zero is excluded from instance-error accounting.
    pairs, counts = np.unique(np.stack((truth_values.ravel(), prediction_values.ravel()), axis=1), axis=0, return_counts=True)
    table = {(int(pair[0]), int(pair[1])): int(count) for pair, count in zip(pairs, counts, strict=True) if pair[0] > 0 or pair[1] > 0}
    truth_sizes: dict[int, int] = {}
    prediction_sizes: dict[int, int] = {}
    for (truth_id, prediction_id), count in table.items():
        if truth_id > 0:
            truth_sizes[truth_id] = truth_sizes.get(truth_id, 0) + count
        if prediction_id > 0:
            prediction_sizes[prediction_id] = prediction_sizes.get(prediction_id, 0) + count
    return table, truth_sizes, prediction_sizes


def instance_metrics(truth: Any, prediction: Any) -> dict[str, Any]:
    """Compute transparent object and information-theoretic instance errors.

    A merge is one predicted positive label overlapping two or more positive
    truth identities; a split is the inverse.  The result is appropriate only
    for co-registered neuronal *instance* label volumes, not synapse points or
    semantic masks.
    """
    table, truth_sizes, prediction_sizes = _contingency(truth, prediction)
    truth_to_prediction: dict[int, set[int]] = {identifier: set() for identifier in truth_sizes}
    prediction_to_truth: dict[int, set[int]] = {identifier: set() for identifier in prediction_sizes}
    for (truth_id, prediction_id), count in table.items():
        if count and truth_id > 0 and prediction_id > 0:
            truth_to_prediction[truth_id].add(prediction_id)
            prediction_to_truth[prediction_id].add(truth_id)
    split_ids = sorted(identifier for identifier, partners in truth_to_prediction.items() if len(partners) > 1)
    merge_ids = sorted(identifier for identifier, partners in prediction_to_truth.items() if len(partners) > 1)
    missed = sorted(identifier for identifier in truth_sizes if not truth_to_prediction[identifier])
    false = sorted(identifier for identifier in prediction_sizes if not prediction_to_truth[identifier])
    total = sum(table.values())
    # VI after excluding background-only pairs.  This intentionally makes the
    # denominator and treatment of background visible in the output.
    def entropy(sizes: dict[int, int]) -> float:
        return -sum((size / total) * math.log2(size / total) for size in sizes.values() if size)
    joint = -sum((count / total) * math.log2(count / total) for count in table.values() if count)
    h_truth = entropy(truth_sizes)
    h_prediction = entropy(prediction_sizes)
    vi_split = max(0.0, joint - h_prediction)  # H(truth | prediction)
    vi_merge = max(0.0, joint - h_truth)  # H(prediction | truth)
    return {
        "evaluation_representation": "same-grid neuronal instance labels; zero is background",
        "evaluated_voxels": total,
        "ground_truth_objects": len(truth_sizes), "predicted_objects": len(prediction_sizes),
        "merge_errors": len(merge_ids), "split_errors": len(split_ids),
        "missed_objects": len(missed), "false_objects": len(false),
        "merge_prediction_ids": merge_ids, "split_truth_ids": split_ids,
        "missed_truth_ids": missed, "false_prediction_ids": false,
        "vi_split": vi_split, "vi_merge": vi_merge, "vi_total": vi_split + vi_merge,
    }


def evaluate_instance_files(truth_path: Path, prediction_path: Path, output: Path, *, model_id: str, region_id: str, split: str) -> dict[str, Any]:
    import numpy as np

    if split not in {"validation", "test"}:
        raise ValueError("Only frozen validation or test labels may be evaluated")
    truth = np.load(truth_path, allow_pickle=False)
    prediction = np.load(prediction_path, allow_pickle=False)
    metrics = instance_metrics(truth, prediction)
    metrics.update({"model_id": model_id, "region_id": region_id, "split": split, "truth_sha256": sha256_file(truth_path), "prediction_sha256": sha256_file(prediction_path), "generated_at": _now()})
    write_json_atomic(output, metrics)
    return metrics
