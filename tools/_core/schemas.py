"""Shared connectomics enums and ToolReceipt schema helpers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class RegionClass(str, Enum):
    """Tissue / tile information class for routing and QA."""

    EMPTY = "EMPTY"
    LOW_INFORMATION = "LOW_INFORMATION"
    NORMAL = "NORMAL"
    HIGH_COMPLEXITY = "HIGH_COMPLEXITY"
    UNCERTAIN = "UNCERTAIN"
    ARTIFACT = "ARTIFACT"
    REPROCESS = "REPROCESS"


class SynapseVerdict(str, Enum):
    """Consensus verdict for a candidate synaptic contact."""

    SUPPORTED = "SUPPORTED"
    PROBABLE = "PROBABLE"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTING = "CONFLICTING"
    REVIEW = "REVIEW"
    REJECTED = "REJECTED"


class SeamAction(str, Enum):
    """Action taken or proposed at a block / tile seam."""

    MATCH = "MATCH"
    SPLIT = "SPLIT"
    MERGE = "MERGE"
    REVIEW = "REVIEW"
    HOLD = "HOLD"


class MorphAnomaly(str, Enum):
    """Morphological anomaly types for axon / dendrite QA."""

    BREAK = "BREAK"
    SPUR = "SPUR"
    LOOP = "LOOP"
    MERGE_ERROR = "MERGE_ERROR"
    SPLIT_ERROR = "SPLIT_ERROR"
    THICKNESS_OUTLIER = "THICKNESS_OUTLIER"
    ENDPOINT_ORPHAN = "ENDPOINT_ORPHAN"
    SELF_TOUCH = "SELF_TOUCH"
    CROSSING = "CROSSING"
    OTHER = "OTHER"


TOOL_RECEIPT_REQUIRED = (
    "tool",
    "timestamp",
    "status",
    "payload",
)


@dataclass
class ToolReceipt:
    """Normalized receipt written by Connectome Project tools."""

    tool: str
    timestamp: str
    status: str = "ok"
    payload: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    region_class: Optional[str] = None
    synapse_verdict: Optional[str] = None
    seam_action: Optional[str] = None
    morph_anomalies: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolReceipt":
        return cls(
            tool=str(data["tool"]),
            timestamp=str(data.get("timestamp") or _utc_now()),
            status=str(data.get("status", "ok")),
            payload=dict(data.get("payload") or {}),
            inputs=list(data.get("inputs") or []),
            outputs=list(data.get("outputs") or []),
            region_class=_opt_enum_str(data.get("region_class"), RegionClass),
            synapse_verdict=_opt_enum_str(data.get("synapse_verdict"), SynapseVerdict),
            seam_action=_opt_enum_str(data.get("seam_action"), SeamAction),
            morph_anomalies=[_enum_str(x, MorphAnomaly) for x in (data.get("morph_anomalies") or [])],
            meta=dict(data.get("meta") or {}),
        )


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _enum_str(value: Any, enum_cls: type[Enum]) -> str:
    if isinstance(value, enum_cls):
        return value.value
    s = str(value)
    try:
        return enum_cls(s).value
    except ValueError:
        # Allow already-canonical names
        for m in enum_cls:
            if m.name == s or m.value == s:
                return m.value
        raise


def _opt_enum_str(value: Any, enum_cls: type[Enum]) -> Optional[str]:
    if value is None or value == "":
        return None
    return _enum_str(value, enum_cls)


def build_tool_receipt(
    tool: str,
    payload: Optional[Mapping[str, Any]] = None,
    *,
    status: str = "ok",
    inputs: Optional[Sequence[str]] = None,
    outputs: Optional[Sequence[str]] = None,
    region_class: Optional[RegionClass | str] = None,
    synapse_verdict: Optional[SynapseVerdict | str] = None,
    seam_action: Optional[SeamAction | str] = None,
    morph_anomalies: Optional[Sequence[MorphAnomaly | str]] = None,
    meta: Optional[Mapping[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> ToolReceipt:
    """Construct a validated ToolReceipt from loosely typed tool output."""
    receipt = ToolReceipt(
        tool=str(tool),
        timestamp=timestamp or _utc_now(),
        status=str(status),
        payload=dict(payload or {}),
        inputs=list(inputs or []),
        outputs=list(outputs or []),
        region_class=_opt_enum_str(region_class, RegionClass),
        synapse_verdict=_opt_enum_str(synapse_verdict, SynapseVerdict),
        seam_action=_opt_enum_str(seam_action, SeamAction),
        morph_anomalies=[_enum_str(x, MorphAnomaly) for x in (morph_anomalies or [])],
        meta=dict(meta or {}),
    )
    validate_tool_receipt(receipt.to_dict())
    return receipt


def validate_tool_receipt(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate required keys and enum fields; return a normalized dict."""
    missing = [k for k in TOOL_RECEIPT_REQUIRED if k not in data]
    if missing:
        raise ValueError(f"ToolReceipt missing keys: {missing}")
    if not str(data["tool"]).strip():
        raise ValueError("ToolReceipt.tool must be non-empty")
    normalized = ToolReceipt.from_dict(data).to_dict()
    return normalized


class GapKind(str, Enum):
    """Completeness gap categories for GapHound."""

    PROCESSED = "processed"
    MISSING = "missing"
    FAILED = "failed"
    STALE = "stale"
    UNVERIFIED = "unverified"
    BOUNDARY_INCOMPLETE = "boundary-incomplete"

