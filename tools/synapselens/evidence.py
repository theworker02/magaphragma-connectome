"""Synapse evidence classification (deterministic rule table)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

try:
    from tools._core import SynapseVerdict
except Exception:  # pragma: no cover
    from enum import Enum

    class SynapseVerdict(str, Enum):
        SUPPORTED = "SUPPORTED"
        PROBABLE = "PROBABLE"
        AMBIGUOUS = "AMBIGUOUS"
        CONFLICTING = "CONFLICTING"
        REVIEW = "REVIEW"
        REJECTED = "REJECTED"


@dataclass(frozen=True)
class CandidateSynapse:
    """Candidate synapse record with multi-source evidence scores."""

    pre_id: str
    post_id: str
    xyz: tuple[float, float, float]
    prediction_confidence: float
    local_image_evidence: float
    segmentation_confidence: float
    boundary_status: str
    morphological_context: float
    model_agreement: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def overall_confidence(self) -> float:
        scores = (
            self.prediction_confidence,
            self.local_image_evidence,
            self.segmentation_confidence,
            self.model_agreement,
            self.morphological_context,
        )
        return sum(scores) / float(len(scores))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["xyz"] = list(self.xyz)
        return d


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def soft_verdict_masses(c: CandidateSynapse) -> dict[str, float]:
    """Deterministic soft masses used for entropy / information gain."""
    pc = _clip01(c.prediction_confidence)
    ie = _clip01(c.local_image_evidence)
    sc = _clip01(c.segmentation_confidence)
    ma = _clip01(c.model_agreement)
    mc = _clip01(c.morphological_context)
    mean = (pc + ie + sc + ma + mc) / 5.0
    conflict = abs(pc - ie) * (1.0 - ma)
    status = (c.boundary_status or "").strip().lower()
    boundary_penalty = 0.0
    if status in {"uncertain", "partial", "crossing", "ambiguous"}:
        boundary_penalty = 0.35
    elif status not in {"clean", "ok", "resolved", ""}:
        boundary_penalty = 0.15

    masses = {
        SynapseVerdict.SUPPORTED.value: max(0.0, mean * ma * (1.0 - conflict) - boundary_penalty),
        SynapseVerdict.PROBABLE.value: max(0.0, mean * 0.85 - conflict * 0.5),
        SynapseVerdict.AMBIGUOUS.value: max(0.0, 0.25 + boundary_penalty + (0.5 - abs(mean - 0.5))),
        SynapseVerdict.CONFLICTING.value: max(0.0, conflict * 1.5),
        SynapseVerdict.REVIEW.value: max(0.0, (1.0 - mc) * 0.6 + (1.0 - mean) * 0.2),
        SynapseVerdict.REJECTED.value: max(0.0, (0.35 - mean) * 2.0) if mean < 0.35 else 0.0,
    }
    total = sum(masses.values()) or 1.0
    return {k: v / total for k, v in masses.items()}


def classify(candidate: CandidateSynapse) -> SynapseVerdict:
    """Apply deterministic rule table -> SynapseVerdict."""
    pc = _clip01(candidate.prediction_confidence)
    ie = _clip01(candidate.local_image_evidence)
    sc = _clip01(candidate.segmentation_confidence)
    ma = _clip01(candidate.model_agreement)
    mc = _clip01(candidate.morphological_context)
    status = (candidate.boundary_status or "").strip().lower()
    mean = (pc + ie + sc + ma + mc) / 5.0
    conflict_gap = abs(pc - ie)

    if pc < 0.20 or ie < 0.15:
        return SynapseVerdict.REJECTED
    if ma < 0.40 and conflict_gap > 0.40:
        return SynapseVerdict.CONFLICTING
    if (
        pc >= 0.75
        and ie >= 0.75
        and sc >= 0.75
        and ma >= 0.80
        and mc >= 0.70
        and status in {"clean", "ok", "resolved"}
    ):
        return SynapseVerdict.SUPPORTED
    if mean >= 0.60 and ma >= 0.55 and conflict_gap <= 0.35:
        return SynapseVerdict.PROBABLE
    if status in {"uncertain", "partial", "crossing", "ambiguous"} or (
        0.40 <= mean < 0.60 and conflict_gap >= 0.25
    ):
        return SynapseVerdict.AMBIGUOUS
    if mc < 0.45 and 0.35 <= mean < 0.70:
        return SynapseVerdict.REVIEW
    return SynapseVerdict.REVIEW


@dataclass
class SynapseLensReport:
    """Aggregate classification + prioritized human review queue."""

    counts: dict[str, int]
    review_queue: list[dict[str, Any]]
    total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "review_queue": list(self.review_queue),
            "total": self.total,
        }


def build_report(
    candidates: list[CandidateSynapse],
    *,
    ranked: list[tuple[CandidateSynapse, SynapseVerdict, float]] | None = None,
) -> SynapseLensReport:
    from .prioritize import prioritize

    ranked = ranked if ranked is not None else prioritize(candidates)
    counts = {v.value: 0 for v in SynapseVerdict}
    queue: list[dict[str, Any]] = []
    for cand, verdict, gain in ranked:
        counts[verdict.value] = counts.get(verdict.value, 0) + 1
        if verdict in {
            SynapseVerdict.AMBIGUOUS,
            SynapseVerdict.CONFLICTING,
            SynapseVerdict.REVIEW,
        }:
            queue.append(
                {
                    "pre_id": cand.pre_id,
                    "post_id": cand.post_id,
                    "xyz": list(cand.xyz),
                    "verdict": verdict.value,
                    "information_gain": round(gain, 8),
                    "overall_confidence": round(cand.overall_confidence(), 8),
                }
            )
    return SynapseLensReport(counts=counts, review_queue=queue, total=len(candidates))
