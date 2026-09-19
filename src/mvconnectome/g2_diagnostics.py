"""Rank frozen-G1 review candidates inside pre-frozen spatial G2 regions.

Candidates are model-assisted navigation aids only. A high G1 affinity plus a
raw image gradient is not a biological boundary decision; only expert review
can create SAME/DIFFERENT affinity supervision.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file, write_json_atomic


def _now() -> str:
    """Return a UTC timestamp for immutable diagnostic queue provenance."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _nms(candidates: list[dict[str, Any]], count: int, minimum_distance: float) -> list[dict[str, Any]]:
    """Keep spatially separated questions without treating scores as truth."""
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda value: value["rank_score"], reverse=True):
        point = np.asarray(candidate["pair_left_zyx"], dtype=float)
        if all(np.linalg.norm(point - np.asarray(item["pair_left_zyx"], dtype=float)) >= minimum_distance for item in selected):
            selected.append(candidate)
            if len(selected) == count:
                break
    return selected


def create_g2_diagnostic_queue(*, cohort_path: Path, region_id: str, inference_dir: Path, output_path: Path,
                               expected_checkpoint_sha256: str, merge_confidence: float = 0.9,
                               candidate_count: int = 12, control_count: int = 6, minimum_distance: float = 12.0) -> dict[str, Any]:
    """Create a frozen raw/G1 review queue without producing labels."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite G2 diagnostic queue: {output_path}")
    if not 0 < merge_confidence < 1 or candidate_count < 1 or control_count < 1 or minimum_distance <= 0:
        raise ValueError("Invalid fixed G2 diagnostic parameters")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if cohort.get("status") != "FROZEN_BEFORE_G1_DIAGNOSTIC_REVIEW":
        raise ValueError("G2 cohort must be frozen before diagnostic ranking")
    region = next((item for item in cohort["regions"] if item["id"] == region_id), None)
    if region is None:
        raise ValueError("Unknown frozen G2 region")
    manifest = json.loads((inference_dir / "inference.json").read_text(encoding="utf-8"))
    if manifest["input"]["sha256"] != region["raw_sha256"] or manifest["input"]["shape_zyx"] != region["shape_zyx"]:
        raise ValueError("G1 inference input does not match frozen raw G2 region")
    if manifest["checkpoint"]["sha256"] != expected_checkpoint_sha256:
        raise ValueError("Diagnostic must use the frozen selected G1 checkpoint")
    affinities = np.load(inference_dir / manifest["affinities"]["file"], mmap_mode="r", allow_pickle=False)
    raw = np.load(region["raw_path"], mmap_mode="r", allow_pickle=False)
    if affinities.shape != (3,) + tuple(raw.shape) or not np.isfinite(affinities).all() or np.any((affinities < 0) | (affinities > 1)):
        raise ValueError("Invalid finite CZYX affinity output")
    raw_float = np.asarray(raw, dtype=np.float32)
    gradients = np.gradient(raw_float)
    gradient = np.sqrt(sum(part * part for part in gradients), dtype=np.float32)
    high_gradient, low_gradient = float(np.quantile(gradient, 0.95)), float(np.quantile(gradient, 0.30))
    contradiction: list[dict[str, Any]] = []
    continuity: list[dict[str, Any]] = []
    for channel in range(3):
        slices = [slice(None), slice(None), slice(None)]; slices[channel] = slice(0, -1)
        for index in np.ndindex(gradient[tuple(slices)].shape):
            right = list(index); right[channel] += 1
            affinity = float(affinities[(channel,) + tuple(right)])
            score = float((gradient[index] + gradient[tuple(right)]) / 2)
            candidate = {"pair_left_zyx": [int(value) for value in index], "pair_right_zyx": [int(value) for value in right],
                         "channel_zyx": channel, "channel_name": ("Z", "Y", "X")[channel], "g1_affinity": affinity,
                         "raw_gradient_score": score}
            if affinity >= merge_confidence and score >= high_gradient:
                contradiction.append({**candidate, "kind": "MERGE_CONFIDENT_RAW_BOUNDARY_CANDIDATE", "rank_score": affinity * score})
            elif affinity >= merge_confidence and score <= low_gradient:
                continuity.append({**candidate, "kind": "MERGE_CONFIDENT_CONTINUITY_CONTROL", "rank_score": affinity * (low_gradient - score + 1e-6)})
    questions = _nms(contradiction, candidate_count, minimum_distance) + _nms(continuity, control_count, minimum_distance)
    value = {
        "schema_version": 1, "id": f"MV-G2-G1-DIAGNOSTIC-{region_id[-1:]}", "created_at": _now(),
        "status": "EXPERT_REVIEW_REQUIRED", "classification": "MACHINE_ASSISTED_REVIEW_NAVIGATION_ONLY",
        "cohort": {"path": str(cohort_path.resolve()), "sha256": sha256_file(cohort_path)},
        "region": {"id": region_id, "role": region["role"], "raw_sha256": region["raw_sha256"], "raw_path": region["raw_path"]},
        "g1_inference": {"path": str(inference_dir.resolve()), "inference_sha256": sha256_file(inference_dir / "inference.json"), "checkpoint_sha256": expected_checkpoint_sha256},
        "parameters": {"merge_confidence": merge_confidence, "raw_gradient_quantiles": {"high": 0.95, "low": 0.30, "high_value": high_gradient, "low_value": low_gradient}, "minimum_distance_voxels": minimum_distance},
        "candidate_counts_before_nms": {"merge_confident_raw_boundary": len(contradiction), "continuity_control": len(continuity)},
        "questions": questions,
        "allowed_decisions": ["SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"],
        "scientific_boundary": "G1 affinity and raw gradients only prioritize inspection. They cannot produce a target label without an identified reviewer decision.",
    }
    write_json_atomic(output_path, value)
    return value
