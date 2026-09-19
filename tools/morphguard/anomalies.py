"""AnomalyMap ? morphology review findings (never definitive biology claims)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Anomaly:
    kind: str
    subject_id: str
    score: float
    evidence: dict[str, Any]
    note: str = "Heuristic morphology flag ? not a definitive biology error."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyMap:
    anomalies: list[Anomaly] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, anomaly: Anomaly) -> None:
        self.anomalies.append(anomaly)

    def review_queue(self, *, min_score: float = 0.35) -> list[dict[str, Any]]:
        items = [a.to_dict() for a in self.anomalies if a.score >= min_score]
        items.sort(key=lambda d: (-d["score"], d["kind"], d["subject_id"]))
        return items

    def kind_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for a in self.anomalies:
            counts[a.kind] = counts.get(a.kind, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": "Flags are heuristic review aids, not definitive biological errors.",
            "n_anomalies": len(self.anomalies),
            "kind_counts": self.kind_counts(),
            "review_queue": self.review_queue(),
            "anomalies": [a.to_dict() for a in self.anomalies],
            "meta": self.meta,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + chr(10), encoding="utf-8")
        return path
