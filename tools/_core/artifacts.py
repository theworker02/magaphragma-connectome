"""Content-addressed artifact store with JSON and npy sidecar receipts."""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

BytesLike = Union[bytes, bytearray, memoryview]
JsonLike = Union[Mapping[str, Any], list, str, int, float, bool, None]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_bytes(data: BytesLike) -> str:
    h = hashlib.sha256()
    h.update(bytes(data))
    return h.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable reference to a content-addressed artifact."""

    sha256: str
    kind: str
    name: str = ""
    media_type: str = "application/octet-stream"
    size_bytes: int = 0
    meta: Mapping[str, Any] = field(default_factory=dict)

    def short(self) -> str:
        return self.sha256[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "kind": self.kind,
            "name": self.name,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            sha256=str(data["sha256"]),
            kind=str(data.get("kind", "blob")),
            name=str(data.get("name", "")),
            media_type=str(data.get("media_type", "application/octet-stream")),
            size_bytes=int(data.get("size_bytes", 0)),
            meta=dict(data.get("meta") or {}),
        )


class ArtifactStore:
    """Content-addressed store: blobs by sha256, receipts under receipts/artifacts/."""

    def __init__(
        self,
        root: Optional[Path] = None,
        *,
        blobs_dir: Optional[Path] = None,
        receipts_dir: Optional[Path] = None,
    ) -> None:
        self.root = Path(root) if root is not None else _project_root()
        self.blobs_dir = (
            Path(blobs_dir) if blobs_dir is not None else (self.root / "derived" / "artifacts")
        )
        self.receipts_dir = (
            Path(receipts_dir)
            if receipts_dir is not None
            else (self.root / "receipts" / "artifacts")
        )
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        self.receipts_dir.mkdir(parents=True, exist_ok=True)

    def _blob_path(self, digest: str) -> Path:
        return self.blobs_dir / digest[:2] / digest[2:4] / digest

    def _receipt_path(self, digest: str) -> Path:
        return self.receipts_dir / f"{digest}.json"

    def _npy_sidecar_path(self, digest: str) -> Path:
        return self.receipts_dir / f"{digest}.npy"

    def has(self, digest: str) -> bool:
        return self._blob_path(digest).is_file()

    def put_bytes(
        self,
        data: BytesLike,
        *,
        kind: str = "blob",
        name: str = "",
        media_type: str = "application/octet-stream",
        meta: Optional[Mapping[str, Any]] = None,
    ) -> ArtifactRef:
        raw = bytes(data)
        digest = sha256_bytes(raw)
        dest = self._blob_path(digest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(raw)
            tmp.replace(dest)
        ref = ArtifactRef(
            sha256=digest,
            kind=kind,
            name=name,
            media_type=media_type,
            size_bytes=len(raw),
            meta=dict(meta or {}),
        )
        self._write_receipt(ref)
        return ref

    def put_json(
        self,
        obj: JsonLike,
        *,
        kind: str = "json",
        name: str = "",
        meta: Optional[Mapping[str, Any]] = None,
    ) -> ArtifactRef:
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return self.put_bytes(
            payload.encode("utf-8"),
            kind=kind,
            name=name,
            media_type="application/json",
            meta=meta,
        )

    def put_npy(
        self,
        array: np.ndarray,
        *,
        kind: str = "ndarray",
        name: str = "",
        meta: Optional[Mapping[str, Any]] = None,
    ) -> ArtifactRef:
        buf = io.BytesIO()
        arr = np.asarray(array)
        np.save(buf, arr, allow_pickle=False)
        raw = buf.getvalue()
        ref = self.put_bytes(
            raw,
            kind=kind,
            name=name,
            media_type="application/x-npy",
            meta={
                **dict(meta or {}),
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
            },
        )
        self._npy_sidecar_path(ref.sha256).write_bytes(raw)
        return ref

    def read_bytes(self, ref_or_digest: Union[ArtifactRef, str]) -> bytes:
        digest = ref_or_digest.sha256 if isinstance(ref_or_digest, ArtifactRef) else str(ref_or_digest)
        path = self._blob_path(digest)
        if not path.is_file():
            raise FileNotFoundError(f"artifact blob missing: {digest}")
        return path.read_bytes()

    def read_json(self, ref_or_digest: Union[ArtifactRef, str]) -> Any:
        return json.loads(self.read_bytes(ref_or_digest).decode("utf-8"))

    def read_npy(self, ref_or_digest: Union[ArtifactRef, str]) -> np.ndarray:
        digest = ref_or_digest.sha256 if isinstance(ref_or_digest, ArtifactRef) else str(ref_or_digest)
        sidecar = self._npy_sidecar_path(digest)
        if sidecar.is_file():
            return np.load(sidecar, allow_pickle=False)
        return np.load(io.BytesIO(self.read_bytes(digest)), allow_pickle=False)

    def read_receipt(self, ref_or_digest: Union[ArtifactRef, str]) -> dict[str, Any]:
        digest = ref_or_digest.sha256 if isinstance(ref_or_digest, ArtifactRef) else str(ref_or_digest)
        path = self._receipt_path(digest)
        if not path.is_file():
            raise FileNotFoundError(f"artifact receipt missing: {digest}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_receipt(self, ref: ArtifactRef) -> Path:
        path = self._receipt_path(ref.sha256)
        try:
            rel = str(self._blob_path(ref.sha256).relative_to(self.root))
        except ValueError:
            rel = str(self._blob_path(ref.sha256))
        rel = rel.replace(chr(92), '/')
        record = {
            "artifact": ref.to_dict(),
            "blob_relpath": rel,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path.write_text(
            json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + chr(10),
            encoding="utf-8",
        )
        return path
