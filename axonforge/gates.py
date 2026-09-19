"""Equivalence / acceptance gates — never hide rejects."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateResult:
    passed: bool
    reason: str
    priority: float = 0.0


def equivalence_gate(candidate_checksum: str, reference_checksum: str, *, tol_name: str = "strict") -> GateResult:
    if candidate_checksum == reference_checksum:
        return GateResult(True, "exact_match", 0.0)
    # Halo-cached tiles may share prefix; still require exact for promotion.
    return GateResult(False, "equivalence_gate_reject", 0.90)


def speed_gate(candidate_seconds: float, reference_seconds: float, min_speedup: float = 1.05) -> GateResult:
    if reference_seconds <= 0 or candidate_seconds <= 0:
        return GateResult(False, "invalid_timing", 1.0)
    speedup = reference_seconds / candidate_seconds
    if speedup >= min_speedup:
        return GateResult(True, f"speedup_{speedup:.3f}", 0.0)
    return GateResult(False, f"no_speedup_{speedup:.3f}", 0.5)
