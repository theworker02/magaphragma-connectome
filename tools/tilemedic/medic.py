"""Diagnose failed tiles and attempt scientifically safe recovery."""
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from _core._receipts import write_tool_receipt

REPO = Path(__file__).resolve().parents[2]


# Scientific parameters that MUST NOT be altered by recovery.
PROTECTED_KEYS = frozenset({
    "checkpoint", "model", "threshold", "precision", "halo",
    "tile_dimensions", "normalization", "algorithm_version",
})


@dataclass
class RecoveryReceipt:
    tile_key: str
    diagnosis: str
    action: str
    success: bool
    safe: bool
    detail: dict = field(default_factory=dict)
    altered_keys: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class TileMedic:
    """Recover failed tiles without quietly changing scientific parameters.

    Safe: lower batch size, clean worker retry, quarantine corrupt input,
    invalidate truncated output, reproduce NaN with backend comparison.
    Unsafe/forbidden: changing model thresholds, checkpoints, or loss params.
    """

    def diagnose(self, failure: dict[str, Any]) -> str:
        msg = (failure.get("error") or failure.get("reason") or "").lower()
        code = (failure.get("code") or "").lower()
        blob = msg + " " + code
        if "oom" in blob or "out of memory" in blob or "hiperroroutofmemory" in blob:
            return "OOM"
        if "corrupt" in blob or "checksum" in blob or "truncated" in blob:
            return "CORRUPT_INPUT"
        if "nan" in blob or "inf" in blob or "nonfinite" in blob:
            return "NAN"
        if "crash" in blob or "segfault" in blob or "worker died" in blob:
            return "WORKER_CRASH"
        if "truncat" in blob or "short read" in blob or "incomplete write" in blob:
            return "OUTPUT_TRUNCATED"
        if failure.get("deterministic") or "assertion" in blob:
            return "DETERMINISTIC_FAILURE"
        return "UNKNOWN"

    def recover(self, failure: dict[str, Any], *, config: dict | None = None) -> RecoveryReceipt:
        config = dict(config or {})
        tile_key = str(failure.get("tile_key") or failure.get("id") or "unknown")
        diagnosis = self.diagnose(failure)
        altered: list[str] = []

        if diagnosis == "OOM":
            # Safe: lower batch only
            old = int(config.get("batch_size", 8))
            new = max(1, old // 2)
            if new != old:
                config["batch_size"] = new
                altered.append("batch_size")
            receipt = RecoveryReceipt(tile_key, diagnosis, "lower_batch_retry", True, True, {"batch_size": new}, altered)

        elif diagnosis == "CORRUPT_INPUT":
            qdir = REPO / "receipts" / "tilemedic" / "quarantine"
            qdir.mkdir(parents=True, exist_ok=True)
            qpath = qdir / f"{tile_key}.json"
            qpath.write_text(json.dumps(failure, indent=2) + chr(10), encoding="utf-8")
            receipt = RecoveryReceipt(tile_key, diagnosis, "quarantine", True, True, {"quarantine": str(qpath)}, [])

        elif diagnosis == "NAN":
            # Reproduce + backend comparison — do not change scientific params
            detail = {
                "reproduce": True,
                "compare_backends": ["eager", "reference"],
                "note": "NaN recovery never mutates thresholds/checkpoints",
            }
            receipt = RecoveryReceipt(tile_key, diagnosis, "reproduce_and_compare", True, True, detail, [])

        elif diagnosis == "WORKER_CRASH":
            receipt = RecoveryReceipt(tile_key, diagnosis, "clean_worker_retry", True, True, {"worker_reset": True}, [])

        elif diagnosis == "OUTPUT_TRUNCATED":
            receipt = RecoveryReceipt(tile_key, diagnosis, "invalidate_and_recompute", True, True, {"invalidate": True}, [])

        elif diagnosis == "DETERMINISTIC_FAILURE":
            bundle = self._diagnostic_bundle(tile_key, failure, config)
            receipt = RecoveryReceipt(tile_key, diagnosis, "STOP_diagnostic_bundle", False, True, {"bundle": str(bundle)}, [])

        else:
            receipt = RecoveryReceipt(tile_key, diagnosis, "manual_review", False, True, {"failure": failure}, [])

        # Enforce: no protected key mutations
        for k in altered:
            if k in PROTECTED_KEYS:
                raise RuntimeError(f"TileMedic refused to alter scientific parameter: {k}")

        write_tool_receipt("tilemedic", receipt.as_dict())
        out = REPO / "receipts" / "tilemedic" / "latest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt.as_dict(), indent=2) + chr(10), encoding="utf-8")
        return receipt

    def _diagnostic_bundle(self, tile_key: str, failure: dict, config: dict) -> Path:
        root = REPO / "receipts" / "tilemedic" / "diagnostics" / tile_key
        root.mkdir(parents=True, exist_ok=True)
        payload = {"tile_key": tile_key, "failure": failure, "config_fingerprint": self._fingerprint(config)}
        # strip protected values into fingerprint only
        (root / "bundle.json").write_text(json.dumps(payload, indent=2) + chr(10), encoding="utf-8")
        return root / "bundle.json"

    def _fingerprint(self, config: dict) -> str:
        sci = {k: config[k] for k in sorted(config) if k in PROTECTED_KEYS}
        raw = json.dumps(sci, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]
