"""Information-gain prioritization for human synapse review."""
from __future__ import annotations

import math
from typing import Any

from .evidence import CandidateSynapse, classify, soft_verdict_masses

try:
    from _core.schemas import SynapseVerdict
except Exception:  # pragma: no cover
    from .evidence import SynapseVerdict  # type: ignore


def _entropy(masses: dict[str, float]) -> float:
    h = 0.0
    for p in masses.values():
        if p > 0.0:
            h -= p * math.log(p, 2)
    return h


def information_gain(candidate: CandidateSynapse) -> float:
    """entropy(verdict soft-mass) * (1 - confidence) * morphological uncertainty."""
    masses = soft_verdict_masses(candidate)
    conf = candidate.overall_confidence()
    morph_uncertainty = 1.0 - max(0.0, min(1.0, float(candidate.morphological_context)))
    return _entropy(masses) * (1.0 - conf) * morph_uncertainty


def prioritize(
    candidates: list[CandidateSynapse],
) -> list[tuple[CandidateSynapse, Any, float]]:
    """Sort candidates by descending information gain (humans review high-gain first)."""
    ranked: list[tuple[CandidateSynapse, Any, float]] = []
    for cand in candidates:
        verdict = classify(cand)
        gain = information_gain(cand)
        ranked.append((cand, verdict, gain))
    ranked.sort(key=lambda t: (-t[2], t[0].pre_id, t[0].post_id, t[0].xyz))
    return ranked