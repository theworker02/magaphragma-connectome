"""ArtifactVet-lite: validate schema/shape/dtype/hash before stage handoff."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]


@dataclass
class VetIssue:
    level: str  # error | warn
    code: str
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class VetReport:
    ok: bool
    path: str = ""
    kind: str = ""
    issues: list[VetIssue] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "path": self.path,
            "kind": self.kind,
            "issues": [i.as_dict() for i in self.issues],
            "meta": self.meta,
        }

    def raise_if_failed(self) -> None:
        if not self.ok:
            msgs = "; ".join(i.message for i in self.issues if i.level == "error")
            raise ValueError(f"ArtifactVet failed: {msgs}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class ArtifactVet:
    """Lightweight gate for npy/json manifests before pipeline handoff."""

    def vet_json(
        self,
        path: PathLike,
        *,
        required_keys: Sequence[str] = (),
        expected_sha256: Optional[str] = None,
    ) -> VetReport:
        p = Path(path)
        issues: list[VetIssue] = []
        meta: dict[str, Any] = {}
        if not p.is_file():
            return VetReport(False, str(p), "json", [VetIssue("error", "missing", f"file not found: {p}")])
        raw = p.read_bytes()
        digest = sha256_bytes(raw)
        meta["sha256"] = digest
        meta["size_bytes"] = len(raw)
        if expected_sha256 and digest != expected_sha256.lower():
            issues.append(VetIssue("error", "hash_mismatch", f"sha256 {digest[:12]} != expected {expected_sha256[:12]}"))
        try:
            text = raw.decode("utf-8")
            obj = json.loads(text)
        except Exception as exc:
            issues.append(VetIssue("error", "json_parse", str(exc)))
            return VetReport(False, str(p), "json", issues, meta)
        if not isinstance(obj, (dict, list)):
            issues.append(VetIssue("warn", "json_root", f"root type {type(obj).__name__}"))
        if required_keys and isinstance(obj, dict):
            missing = [k for k in required_keys if k not in obj]
            if missing:
                issues.append(VetIssue("error", "schema", f"missing keys: {missing}"))
            meta["keys"] = sorted(obj.keys()) if isinstance(obj, dict) else []
        ok = not any(i.level == "error" for i in issues)
        return VetReport(ok, str(p), "json", issues, meta)

    def vet_npy(
        self,
        path: PathLike,
        *,
        expected_shape: Optional[Sequence[int]] = None,
        expected_dtype: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        ndim: Optional[int] = None,
        allow_empty: bool = False,
    ) -> VetReport:
        p = Path(path)
        issues: list[VetIssue] = []
        meta: dict[str, Any] = {}
        if not p.is_file():
            return VetReport(False, str(p), "npy", [VetIssue("error", "missing", f"file not found: {p}")])
        digest = sha256_file(p)
        meta["sha256"] = digest
        meta["size_bytes"] = p.stat().st_size
        if expected_sha256 and digest != expected_sha256.lower():
            issues.append(VetIssue("error", "hash_mismatch", f"sha256 {digest[:12]} != expected {expected_sha256[:12]}"))
        try:
            arr = np.load(p, allow_pickle=False)
        except Exception as exc:
            issues.append(VetIssue("error", "npy_load", str(exc)))
            return VetReport(False, str(p), "npy", issues, meta)
        meta["shape"] = list(arr.shape)
        meta["dtype"] = str(arr.dtype)
        meta["nbytes"] = int(arr.nbytes)
        if ndim is not None and arr.ndim != ndim:
            issues.append(VetIssue("error", "ndim", f"ndim {arr.ndim} != {ndim}"))
        if expected_shape is not None and tuple(arr.shape) != tuple(int(x) for x in expected_shape):
            issues.append(VetIssue("error", "shape", f"shape {arr.shape} != {tuple(expected_shape)}"))
        if expected_dtype is not None and str(arr.dtype) != str(expected_dtype):
            issues.append(VetIssue("error", "dtype", f"dtype {arr.dtype} != {expected_dtype}"))
        if not allow_empty and arr.size == 0:
            issues.append(VetIssue("error", "empty", "array is empty"))
        if not np.isfinite(arr.astype(np.float64, copy=False)).all() if np.issubdtype(arr.dtype, np.floating) else True:
            # only check float arrays
            if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                issues.append(VetIssue("error", "nonfinite", "array contains NaN/Inf"))
        ok = not any(i.level == "error" for i in issues)
        return VetReport(ok, str(p), "npy", issues, meta)

    def vet_manifest(
        self,
        path: PathLike,
        *,
        required_keys: Sequence[str] = ("schema", "items"),
        expected_sha256: Optional[str] = None,
    ) -> VetReport:
        return self.vet_json(path, required_keys=required_keys, expected_sha256=expected_sha256)

    def vet_handoff(
        self,
        *,
        json_path: Optional[PathLike] = None,
        npy_path: Optional[PathLike] = None,
        json_keys: Sequence[str] = (),
        npy_shape: Optional[Sequence[int]] = None,
        npy_dtype: Optional[str] = None,
    ) -> VetReport:
        """Combined gate used by GapHound / NeuroCache stage handoff."""
        issues: list[VetIssue] = []
        meta: dict[str, Any] = {"parts": []}
        ok = True
        paths = []
        if json_path is not None:
            r = self.vet_json(json_path, required_keys=json_keys)
            meta["parts"].append(r.as_dict())
            issues.extend(r.issues)
            ok = ok and r.ok
            paths.append(str(json_path))
        if npy_path is not None:
            r = self.vet_npy(npy_path, expected_shape=npy_shape, expected_dtype=npy_dtype)
            meta["parts"].append(r.as_dict())
            issues.extend(r.issues)
            ok = ok and r.ok
            paths.append(str(npy_path))
        if not paths:
            issues.append(VetIssue("error", "no_input", "vet_handoff requires json_path and/or npy_path"))
            ok = False
        return VetReport(ok, "+".join(paths), "handoff", issues, meta)


def vet_before_cache_put(payload_path: PathLike, identity_path: PathLike) -> VetReport:
    """Hook NeuroCache can call before put()."""
    p = Path(payload_path)
    if p.suffix.lower() == ".npy":
        return ArtifactVet().vet_handoff(
            json_path=identity_path,
            npy_path=p,
            json_keys=("source_volume_hash", "model", "checkpoint"),
        )
    return ArtifactVet().vet_json(
        identity_path,
        required_keys=("source_volume_hash", "model", "checkpoint"),
    )


def vet_gaphound_scan(path: PathLike) -> VetReport:
    """Hook GapHound can call before consuming a scan receipt."""
    return ArtifactVet().vet_json(
        path,
        required_keys=("shape_zyx", "tile_zyx", "regions", "counts"),
    )
