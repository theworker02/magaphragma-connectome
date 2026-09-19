"""Non-negotiable integrity gate for builds and scientific releases."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .datasets import validate_registry
from .io import read_json, sha256_file


BIOLOGICAL_COLLECTIONS = ("volumes", "segments", "neurons", "synapses", "annotations", "connections")


def _items(root: Path, name: str) -> list[dict[str, Any]]:
    """Load one optional biological-registry collection as JSON object records."""
    path = root / "biological" / f"{name}.json"
    return read_json(path).get(name, []) if path.exists() else []


def _coordinates_valid(value: Any) -> bool:
    """Recognize a finite coordinate triple without accepting booleans as numbers."""
    return isinstance(value, list) and len(value) == 3 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value)


def verify(root: Path, registry_path: Path) -> list[str]:
    """Return every provenance or biological-release invariant violation found."""
    errors = [f"dataset registry: {error}" for error in validate_registry(registry_path)]
    entities = {name: _items(root, name) for name in (*BIOLOGICAL_COLLECTIONS, "evidence")}
    identifiers = {name: {item.get("id") for item in items} for name, items in entities.items()}
    evidence_ids = identifiers["evidence"]
    evidence_by_id = {item.get("id"): item for item in entities["evidence"]}
    dataset_ids = {source["id"] for source in read_json(registry_path)["sources"]}
    for evidence in entities["evidence"]:
        identifier = evidence.get("id", "<missing>")
        if not isinstance(identifier, str) or not identifier.startswith("MV-EV-"):
            errors.append(f"evidence:{identifier}: invalid identifier")
        if evidence.get("dataset_id") not in dataset_ids:
            errors.append(f"evidence:{identifier}: missing dataset attribution {evidence.get('dataset_id')}")
        if evidence.get("coordinates_nm_xyz") is not None and not _coordinates_valid(evidence["coordinates_nm_xyz"]):
            errors.append(f"evidence:{identifier}: invalid coordinates_nm_xyz")
        if evidence.get("status") in {"MACHINE_PREDICTED", "MACHINE_SEGMENTED"} and not evidence.get("model_hash"):
            errors.append(f"evidence:{identifier}: machine provenance lacks model hash")
    for name in BIOLOGICAL_COLLECTIONS:
        for item in entities[name]:
            identifier = item.get("id", "<missing>")
            if not isinstance(identifier, str) or not identifier.startswith("MV-") or identifier.startswith("SYN-"):
                errors.append(f"{name}:{identifier}: synthetic or invalid biological identifier")
            if not item.get("evidence_ids"):
                errors.append(f"{name}:{identifier}: missing evidence")
            for evidence_id in item.get("evidence_ids", []):
                if evidence_id not in evidence_ids:
                    errors.append(f"{name}:{identifier}: orphan evidence {evidence_id}")
            if item.get("dataset_ids") is None:
                errors.append(f"{name}:{identifier}: missing dataset attribution list")
            for dataset_id in item.get("dataset_ids", []):
                if dataset_id not in dataset_ids:
                    errors.append(f"{name}:{identifier}: missing dataset attribution {dataset_id}")
            if item.get("status") == "MANUALLY_VERIFIED" and item.get("review_state") not in {"FIRST_PASS", "SECOND_PASS", "EXPERT_REVIEWED", "LOCKED_RELEASE"}:
                errors.append(f"{name}:{identifier}: impossible verified status without human review")
            if item.get("status") == "MANUALLY_VERIFIED" and not any(evidence_by_id.get(evidence_id, {}).get("status") == "MANUALLY_VERIFIED" for evidence_id in item.get("evidence_ids", [])):
                errors.append(f"{name}:{identifier}: verified claim lacks manually verified evidence")
            for field in ("coordinates_nm_xyz",):
                if field in item and not _coordinates_valid(item[field]):
                    errors.append(f"{name}:{identifier}: invalid {field}")
    for synapse in entities["synapses"]:
        identifier = synapse.get("id", "<missing>")
        for target in (synapse.get("pre_neuron_id"), synapse.get("post_neuron_id")):
            if target is not None and target not in identifiers["neurons"]:
                errors.append(f"synapses:{identifier}: orphan neuron {target}")
        for segment in synapse.get("segment_ids", []):
            if segment not in identifiers["segments"]:
                errors.append(f"synapses:{identifier}: orphan segment {segment}")
    for connection in entities["connections"]:
        identifier = connection.get("id", "<missing>")
        for target in (connection.get("pre_neuron_id"), connection.get("post_neuron_id")):
            if target not in identifiers["neurons"]:
                errors.append(f"connections:{identifier}: orphan neuron {target}")
        for synapse in connection.get("synapse_ids", []):
            if synapse not in identifiers["synapses"]:
                errors.append(f"connections:{identifier}: orphan synapse {synapse}")
    for source in read_json(registry_path)["sources"]:
        bundled = source.get("local_file")
        if bundled:
            file_path = root / bundled
            if not source.get("redistribution_permitted"):
                errors.append(f"dataset:{source['id']}: bundled file without redistribution permission")
            elif not file_path.is_file() or sha256_file(file_path) != source.get("sha256"):
                errors.append(f"dataset:{source['id']}: missing or hash-mismatched bundled file")
    registrations = root / "datasets" / "registrations"
    if registrations.exists():
        for registration in registrations.glob("*.json"):
            record = read_json(registration)
            artifact = record.get("artifact", {})
            artifact_path = Path(artifact.get("path", ""))
            if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.get("sha256"):
                errors.append(f"registration:{registration.name}: missing or hash-mismatched local artifact")
    return errors
