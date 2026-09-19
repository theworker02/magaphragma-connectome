"""Durable I/O primitives shared by provenance-bearing workflows.

Inputs are checksummed, JSON is read explicitly, and JSON writes are atomic so
an interrupted local run cannot leave a half-written scientific receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally, retaining constant memory for large volumes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object, rejecting unexpected top-level JSON values."""
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    """Replace a JSON artifact atomically after its full serialized value exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)
