"""Versionable provenance reports; intentionally useful even for an empty biological registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, write_json_atomic
from .repository import ConnectomeRepository


def provenance_report(root: Path, registry_path: Path, output_path: Path) -> Path:
    registry = read_json(registry_path)
    repository = ConnectomeRepository(root)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "project": "Vigilia Connectome",
        "organism": "Megaphragma viggianii",
        "registry_sha256": sha256_file(registry_path),
        "sources": [{
            "id": item["id"], "doi": item.get("doi"), "license_status": item.get("license_status"),
            "access_status": item.get("access_status"), "citation": item.get("citation"),
        } for item in registry["sources"]],
        "biological_registry": repository.status(),
        "source_audit_evidence_count": len(read_json(root / "provenance" / "evidence.json").get("evidence", [])) if (root / "provenance" / "evidence.json").exists() else 0,
        "interpretation": "Counts are not coverage measures. Absence of records is represented as an empty dataset, not missing values or synthetic samples.",
        "reproduction": {"command": "vigilia verify && vigilia provenance-report", "input_registry": str(registry_path)},
    }
    write_json_atomic(output_path, report)
    return output_path
