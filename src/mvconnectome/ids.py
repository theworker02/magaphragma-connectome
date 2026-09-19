"""Stable public identifier generation. Counters are sequence values, never database row ids."""

from __future__ import annotations


PREFIXES = {
    "neuron": "MV-N-", "synapse": "MV-SYN-", "segment": "MV-SEG-", "annotation": "MV-ANN-",
    "evidence": "MV-EV-", "connection": "MV-CON-", "volume": "MV-VOL-", "release": "MV-REL-",
}


def stable_id(kind: str, serial: int) -> str:
    if kind not in PREFIXES:
        raise ValueError(f"Unsupported identifier kind: {kind}")
    if serial < 1:
        raise ValueError("Identifier serial must be positive")
    width = 8 if kind in {"synapse", "segment", "annotation", "connection"} else 6
    return f"{PREFIXES[kind]}{serial:0{width}d}"
