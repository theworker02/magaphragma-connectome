"""Domain objects that prohibit unproven biological claims by construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
try:  # Python 3.11+
    from enum import StrEnum
except ImportError:  # pinned ELF environment is intentionally Python 3.10
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from typing import Any


class EvidenceStatus(StrEnum):
    """How directly an assertion is supported; values never imply eligibility alone."""
    SOURCE_ANNOTATED = "SOURCE_ANNOTATED"
    MANUALLY_VERIFIED = "MANUALLY_VERIFIED"
    MACHINE_PREDICTED = "MACHINE_PREDICTED"
    MACHINE_SEGMENTED = "MACHINE_SEGMENTED"
    LITERATURE_DERIVED = "LITERATURE_DERIVED"
    INFERRED = "INFERRED"
    DISPUTED = "DISPUTED"
    UNKNOWN = "UNKNOWN"


class ReviewState(StrEnum):
    """Recorded review maturity, kept separate from a source or model evidence status."""
    UNREVIEWED = "UNREVIEWED"
    MACHINE_ONLY = "MACHINE_ONLY"
    FIRST_PASS = "FIRST_PASS"
    SECOND_PASS = "SECOND_PASS"
    EXPERT_REVIEWED = "EXPERT_REVIEWED"
    CONFLICTED = "CONFLICTED"
    LOCKED_RELEASE = "LOCKED_RELEASE"


class CellTypeStatus(StrEnum):
    """Strength of a cell-type assignment; unknown remains an explicit valid state."""
    PUBLISHED_TYPE = "PUBLISHED_TYPE"
    EXPERT_ASSIGNED = "EXPERT_ASSIGNED"
    MORPHOLOGY_CANDIDATE = "MORPHOLOGY_CANDIDATE"
    CONNECTIVITY_CLUSTER = "CONNECTIVITY_CLUSTER"
    UNKNOWN = "UNKNOWN"


def require_identifier(value: str, prefix: str) -> str:
    """Validate a non-empty stable identifier in the namespace required by a field."""
    if not value.startswith(prefix) or len(value) <= len(prefix):
        raise ValueError(f"Expected stable identifier beginning {prefix!r}, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Evidence:
    """Auditable support for one biological assertion or derived object."""

    id: str
    status: EvidenceStatus
    dataset_id: str
    volume_id: str | None
    coordinates_nm_xyz: tuple[float, float, float] | None
    source_url: str | None
    publication_doi: str | None
    method: str
    confidence: float | None = None
    model_hash: str | None = None
    reviewer: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        require_identifier(self.id, "MV-EV-")
        require_identifier(self.dataset_id, "MV-")
        if self.volume_id is not None:
            require_identifier(self.volume_id, "MV-")
        if self.coordinates_nm_xyz is not None and len(self.coordinates_nm_xyz) != 3:
            raise ValueError("Evidence coordinates must be an xyz triplet in nanometres")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Evidence confidence must be between 0 and 1")
        if self.status in {EvidenceStatus.MACHINE_PREDICTED, EvidenceStatus.MACHINE_SEGMENTED} and not self.model_hash:
            raise ValueError("Machine evidence requires an immutable model hash")


@dataclass(frozen=True, slots=True)
class Synapse:
    """A concrete pre/post site whose evidence and human-review state are retained."""
    id: str
    evidence_id: str
    pre_neuron_id: str | None
    post_neuron_id: str | None
    segment_ids: tuple[str, ...]
    coordinates_nm_xyz: tuple[float, float, float]
    status: EvidenceStatus
    review_state: ReviewState
    confidence: float | None = None

    def __post_init__(self) -> None:
        require_identifier(self.id, "MV-SYN-")
        require_identifier(self.evidence_id, "MV-EV-")
        if self.pre_neuron_id is not None:
            require_identifier(self.pre_neuron_id, "MV-N-")
        if self.post_neuron_id is not None:
            require_identifier(self.post_neuron_id, "MV-N-")
        if not self.segment_ids:
            raise ValueError("Synapses require one or more source segments")
        for segment_id in self.segment_ids:
            require_identifier(segment_id, "MV-SEG-")
        if len(self.coordinates_nm_xyz) != 3:
            raise ValueError("Synapse coordinates must be xyz nanometres")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Synapse confidence must be between 0 and 1")
        if self.status is EvidenceStatus.MANUALLY_VERIFIED and self.review_state not in {
            ReviewState.FIRST_PASS, ReviewState.SECOND_PASS, ReviewState.EXPERT_REVIEWED, ReviewState.LOCKED_RELEASE
        }:
            raise ValueError("Verified synapses require a recorded human review state")


@dataclass(frozen=True, slots=True)
class Connection:
    """A derived directed relation backed by concrete synapse and evidence IDs."""
    id: str
    pre_neuron_id: str
    post_neuron_id: str
    synapse_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    status: EvidenceStatus
    review_state: ReviewState

    def __post_init__(self) -> None:
        require_identifier(self.id, "MV-CON-")
        require_identifier(self.pre_neuron_id, "MV-N-")
        require_identifier(self.post_neuron_id, "MV-N-")
        if not self.synapse_ids or not self.evidence_ids:
            raise ValueError("Connections require concrete synapse and evidence identifiers")
        for item in self.synapse_ids:
            require_identifier(item, "MV-SYN-")
        for item in self.evidence_ids:
            require_identifier(item, "MV-EV-")


@dataclass(frozen=True, slots=True)
class VolumeMetadata:
    """Authority-supplied physical and provenance metadata for an ingested volume."""
    dataset_id: str
    specimen_id: str
    format: str
    voxel_size_nm: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]
    coordinate_frame: str
    origin_nm_xyz: tuple[float, float, float]
    source_url: str
    source_sha256: str
    evidence_status: EvidenceStatus
    notes: str | None = None

    def __post_init__(self) -> None:
        require_identifier(self.dataset_id, "MV-")
        if len(self.voxel_size_nm) != 3 or any(value <= 0 for value in self.voxel_size_nm):
            raise ValueError("Voxel size must contain three positive values in nm")
        if len(self.shape_zyx) != 3 or any(value <= 0 for value in self.shape_zyx):
            raise ValueError("Volume shape must contain three positive zyx dimensions")
        if len(self.origin_nm_xyz) != 3:
            raise ValueError("Volume origin must be an xyz triplet in nm")
        if not self.coordinate_frame.strip():
            raise ValueError("A named coordinate frame is required")
        if len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256.lower()):
            raise ValueError("A 64-character SHA-256 source checksum is required")


@dataclass(frozen=True, slots=True)
class BiologicalEntity:
    """Shared invariant for release-eligible entities.

    Synthetic fixtures intentionally use the separate `SYN-` namespace and never instantiate this
    type. An entity can therefore not accidentally gain release eligibility by changing a label.
    """

    id: str
    evidence_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    status: EvidenceStatus
    review_state: ReviewState

    def validate_biological(self, prefix: str) -> None:
        require_identifier(self.id, prefix)
        if self.id.startswith("SYN-"):
            raise ValueError("Synthetic identifiers are not biological entities")
        if not self.evidence_ids or not self.dataset_ids:
            raise ValueError("Biological entities require evidence and dataset attribution")
        for evidence_id in self.evidence_ids:
            require_identifier(evidence_id, "MV-EV-")
        for dataset_id in self.dataset_ids:
            require_identifier(dataset_id, "MV-")


@dataclass(frozen=True, slots=True)
class Volume(BiologicalEntity):
    """An evidence-backed biological volume record, distinct from cached imagery."""
    coordinates_nm_xyz: tuple[float, float, float]
    voxel_size_nm: tuple[float, float, float]

    def __post_init__(self) -> None:
        self.validate_biological("MV-VOL-")
        if len(self.coordinates_nm_xyz) != 3 or len(self.voxel_size_nm) != 3 or any(v <= 0 for v in self.voxel_size_nm):
            raise ValueError("Volumes require valid xyz coordinates and positive voxel size")


@dataclass(frozen=True, slots=True)
class Segment(BiologicalEntity):
    """An evidence-backed segment record; machine labels alone do not instantiate it."""
    volume_id: str
    coordinates_nm_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        self.validate_biological("MV-SEG-")
        require_identifier(self.volume_id, "MV-VOL-")
        if len(self.coordinates_nm_xyz) != 3:
            raise ValueError("Segments require valid xyz coordinates")


@dataclass(frozen=True, slots=True)
class Neuron(BiologicalEntity):
    """A biological neuron claim that must retain supporting segments and evidence."""
    segment_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        self.validate_biological("MV-N-")
        if not self.segment_ids:
            raise ValueError("Neurons must reference one or more evidence-backed segments")
        for segment_id in self.segment_ids:
            require_identifier(segment_id, "MV-SEG-")


@dataclass(frozen=True, slots=True)
class Annotation(BiologicalEntity):
    """A human-readable assertion attached to an evidence-backed biological target."""
    target_id: str
    coordinates_nm_xyz: tuple[float, float, float] | None
    text: str

    def __post_init__(self) -> None:
        self.validate_biological("MV-ANN-")
        if not self.target_id.startswith("MV-") or not self.text.strip():
            raise ValueError("Annotations require a biological target and non-empty text")
        if self.coordinates_nm_xyz is not None and len(self.coordinates_nm_xyz) != 3:
            raise ValueError("Annotation coordinates must be xyz")


def as_json(value: Any) -> Any:
    """Convert first-party domain values to standard JSON-compatible data."""
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [as_json(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {key: as_json(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [as_json(item) for item in value]
    return value


def evidence_from_dict(value: dict[str, Any]) -> Evidence:
    """Deserialize a stored evidence object through the same invariant checks."""
    return Evidence(
        id=value["id"], status=EvidenceStatus(value["status"]), dataset_id=value["dataset_id"],
        volume_id=value.get("volume_id"), coordinates_nm_xyz=tuple(value["coordinates_nm_xyz"]) if value.get("coordinates_nm_xyz") else None,
        source_url=value.get("source_url"), publication_doi=value.get("publication_doi"), method=value["method"],
        confidence=value.get("confidence"), model_hash=value.get("model_hash"), reviewer=value.get("reviewer"), notes=value.get("notes"),
    )


def synapse_from_dict(value: dict[str, Any]) -> Synapse:
    """Deserialize a stored synapse object without bypassing review constraints."""
    return Synapse(
        id=value["id"], evidence_id=value["evidence_id"], pre_neuron_id=value.get("pre_neuron_id"), post_neuron_id=value.get("post_neuron_id"),
        segment_ids=tuple(value["segment_ids"]), coordinates_nm_xyz=tuple(value["coordinates_nm_xyz"]), status=EvidenceStatus(value["status"]),
        review_state=ReviewState(value["review_state"]), confidence=value.get("confidence"),
    )
