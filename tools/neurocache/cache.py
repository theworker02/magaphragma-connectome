"""On-disk semantic computation cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .identity import ScientificIdentity

try:
    from tools._core import write_tool_receipt
except Exception:  # pragma: no cover
    write_tool_receipt = None  # type: ignore


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "receipts" / "neurocache"


class NeuroCache:
    """Semantic cache: blobs/<id_hash>/{manifest.json,payload}."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else _default_root()
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)

    def _blob_dir(self, id_hash: str) -> Path:
        return self.blobs / id_hash

    def put(
        self,
        identity: ScientificIdentity,
        payload: Any,
        *,
        payload_name: str = "payload.json",
    ) -> dict[str, Any]:
        id_hash = identity.identity_hash()
        bdir = self._blob_dir(id_hash)
        bdir.mkdir(parents=True, exist_ok=True)

        if isinstance(payload, (bytes, bytearray)):
            payload_path = bdir / "payload.bin"
            payload_path.write_bytes(bytes(payload))
            checksum = hashlib.sha256(bytes(payload)).hexdigest()
            payload_kind = "bytes"
            payload_file = payload_path.name
        else:
            payload_path = bdir / payload_name
            text = json.dumps(payload, indent=2, sort_keys=True) + chr(10)
            payload_path.write_text(text, encoding="utf-8", newline="\n")
            checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
            payload_kind = "json"
            payload_file = payload_path.name

        manifest = {
            "identity": identity.to_dict(),
            "identity_hash": id_hash,
            "payload_file": payload_file,
            "payload_kind": payload_kind,
            "checksum_sha256": checksum,
        }
        (bdir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + chr(10),
            encoding="utf-8",
            newline="\n",
        )
        if write_tool_receipt is not None:
            write_tool_receipt("neurocache", {"event": "put", "identity_hash": id_hash})
        return manifest

    def lookup(self, identity: ScientificIdentity) -> dict[str, Any] | None:
        """Return exact compatible artifact manifest or None."""
        id_hash = identity.identity_hash()
        man_path = self._blob_dir(id_hash) / "manifest.json"
        if not man_path.exists():
            return None
        artifact = json.loads(man_path.read_text(encoding="utf-8"))
        if not self.validate(artifact, identity=identity):
            return None
        return artifact

    def get(self, identity: ScientificIdentity) -> Any | None:
        artifact = self.lookup(identity)
        if artifact is None:
            return None
        path = self._blob_dir(artifact["identity_hash"]) / artifact["payload_file"]
        if artifact.get("payload_kind") == "bytes":
            return path.read_bytes()
        return json.loads(path.read_text(encoding="utf-8"))

    def validate(
        self,
        artifact: dict[str, Any],
        *,
        identity: ScientificIdentity | None = None,
    ) -> bool:
        """Checksum + identity match."""
        id_hash = artifact.get("identity_hash")
        if not id_hash:
            return False
        bdir = self._blob_dir(str(id_hash))
        payload_path = bdir / artifact.get("payload_file", "payload.json")
        if not payload_path.exists():
            return False

        data = payload_path.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        if checksum != artifact.get("checksum_sha256"):
            return False

        stored_identity = artifact.get("identity") or {}
        try:
            stored = ScientificIdentity.from_dict(stored_identity)
        except Exception:
            return False
        if stored.identity_hash() != id_hash:
            return False
        if identity is not None and identity.identity_hash() != id_hash:
            return False
        return True

    def stats(self) -> dict[str, Any]:
        blobs = [p for p in self.blobs.iterdir() if p.is_dir()] if self.blobs.exists() else []
        total_bytes = 0
        for b in blobs:
            for f in b.rglob("*"):
                if f.is_file():
                    total_bytes += f.stat().st_size
        return {
            "blob_count": len(blobs),
            "total_bytes": total_bytes,
            "root": str(self.root),
        }
