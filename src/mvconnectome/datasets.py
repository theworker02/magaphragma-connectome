"""Dataset registry validation and conservative resumable download support."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

from .io import read_json, sha256_file


def load_registry(path: Path) -> dict:
    """Load the declared source registry before a workflow consumes its records."""
    registry = read_json(path)
    if registry.get("schema_version") != "1.0" or not isinstance(registry.get("sources"), list):
        raise ValueError("Registry must have schema_version 1.0 and a sources array")
    identifiers = [item.get("id") for item in registry["sources"]]
    if len(identifiers) != len(set(identifiers)) or any(not value for value in identifiers):
        raise ValueError("Every dataset source needs a unique id")
    return registry


def validate_registry(path: Path) -> list[str]:
    """List every missing provenance field required for safe source handling."""
    registry = load_registry(path)
    issues: list[str] = []
    required = {"id", "name", "access_status", "license_status", "landing_page", "citation"}
    for source in registry["sources"]:
        missing = sorted(required - source.keys())
        if missing:
            issues.append(f"{source.get('id', '<unknown>')}: missing {', '.join(missing)}")
        if source.get("download_approved"):
            for key in ("download_url", "sha256", "license", "expected_filename"):
                if not source.get(key):
                    issues.append(f"{source['id']}: approved downloads require {key}")
    return issues


def download_source(registry_path: Path, source_id: str, destination: Path) -> Path:
    """Resume only an approved, checksummed source download to local storage."""
    source = next((item for item in load_registry(registry_path)["sources"] if item["id"] == source_id), None)
    if source is None:
        raise KeyError(f"Unknown source: {source_id}")
    if not source.get("download_approved"):
        raise PermissionError("Download is blocked: access, rights, immutable artifact URL, and checksum must be verified first")
    url, expected = source["download_url"], source["sha256"].lower()
    expected_name = source.get("expected_filename")
    if expected_name and destination.name != expected_name:
        raise ValueError(f"Refusing unexpected destination filename; expected {expected_name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    digest = hashlib.sha256()
    if offset:
        with partial.open("rb") as existing:
            while block := existing.read(1024 * 1024):
                digest.update(block)
    with urlopen(request, timeout=60) as response:
        if offset and response.status != 206:
            raise ValueError("Server did not honor byte-range resume; partial file retained and not modified")
        with partial.open("ab") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("Download checksum mismatch; partial file retained for inspection/resume")
    partial.replace(destination)
    return destination
