"""Release-oriented JSON repository with evidence joins for API consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json


class ConnectomeRepository:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _collection(self, name: str) -> list[dict[str, Any]]:
        path = self.root / "biological" / f"{name}.json"
        if not path.exists():
            return []
        value = read_json(path)
        entries = value.get(name, [])
        if not isinstance(entries, list):
            raise ValueError(f"Invalid collection {path}")
        return entries

    def status(self) -> dict[str, Any]:
        return {
            "scientific_release": None,
            "dataset_state": "EMPTY",
            "message": "No biological reconstruction release has been published.",
            "counts": {name: len(self._collection(name)) for name in ("volumes", "segments", "neurons", "synapses", "annotations", "connections")},
        }

    def _by_id(self, name: str, identifier: str) -> dict[str, Any] | None:
        return next((item for item in self._collection(name) if item.get("id") == identifier), None)

    def evidence(self, identifier: str) -> dict[str, Any] | None:
        source_audit = self.root / "provenance" / "evidence.json"
        if source_audit.exists():
            entries = read_json(source_audit).get("evidence", [])
            found = next((item for item in entries if item.get("id") == identifier), None)
            if found is not None:
                return found
        return self._by_id("evidence", identifier)

    def source_dataset(self, identifier: str) -> dict[str, Any] | None:
        """Return auditable source metadata without implying a biological import occurred."""
        registry_path = self.root / "datasets" / "registry.json"
        if not registry_path.exists():
            return None
        sources = read_json(registry_path).get("sources", [])
        return next((source for source in sources if source.get("id") == identifier), None)

    def neuron_with_evidence(self, identifier: str) -> dict[str, Any] | None:
        neuron = self._by_id("neurons", identifier)
        if neuron is None:
            return None
        evidence = [item for item in (self.evidence(eid) for eid in neuron.get("evidence_ids", [])) if item]
        segments = [item for item in (self._by_id("segments", sid) for sid in neuron.get("segment_ids", [])) if item]
        return {"entity": neuron, "evidence": evidence, "segments": segments}

    def connection_trace(self, pre_id: str, post_id: str) -> dict[str, Any] | None:
        connection = next((item for item in self._collection("connections") if item.get("pre_neuron_id") == pre_id and item.get("post_neuron_id") == post_id), None)
        if connection is None:
            return None
        synapses = [item for item in (self._by_id("synapses", sid) for sid in connection.get("synapse_ids", [])) if item]
        evidence = [item for item in (self.evidence(eid) for eid in connection.get("evidence_ids", [])) if item]
        segment_ids = {segment_id for synapse in synapses for segment_id in synapse.get("segment_ids", [])}
        segments = [item for item in (self._by_id("segments", sid) for sid in sorted(segment_ids)) if item]
        volumes = [item for item in (self._by_id("volumes", segment.get("volume_id", "")) for segment in segments) if item]
        return {"connection": connection, "synapses": synapses, "segments": segments, "volumes": volumes, "evidence": evidence}

    def derived_segments(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for path in (self.root / "derived" / "segmentation").glob("**/segments.json"):
            found.extend(read_json(path).get("segments", []))
        return found

    def derived_segment(self, identifier: str) -> dict[str, Any] | None:
        return next((item for item in self.derived_segments() if item.get("id") == identifier), None)
