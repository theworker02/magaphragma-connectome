"""Evidence-scored split/merge assistant — not calibrated biological probabilities."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from _core._receipts import write_tool_receipt

REPO = Path(__file__).resolve().parents[2]


@dataclass
class BranchEvidence:
    name: str
    supports: str  # MERGE | SPLIT | NEUTRAL
    score: float
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BranchVerdict:
    decision: str
    merge_evidence: float
    split_evidence: float
    strongest: list[str] = field(default_factory=list)
    conflicting: list[str] = field(default_factory=list)
    evidence: list[BranchEvidence] = field(default_factory=list)
    source_region: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["note"] = "Scores are evidence weights, not calibrated biological probabilities."
        return d

    def format_summary(self) -> str:
        lines = [
            f"Decision: {self.decision}",
            "",
            f"MERGE evidence: {self.merge_evidence:.2f}",
            f"SPLIT evidence: {self.split_evidence:.2f}",
            "",
            "Strongest evidence:",
        ]
        for s in self.strongest:
            lines.append(f"+ {s}")
        if self.conflicting:
            lines.append("")
            lines.append("Conflicting evidence:")
            for c in self.conflicting:
                lines.append(f"- {c}")
        lines += ["", f"Source region: {self.source_region}", "", "Note: evidence scores, not biological probabilities."]
        return chr(10).join(lines)


def _pca_axis(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0])
    c = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    v = vt[0]
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0])


class BranchJudge:
    def judge(
        self,
        a_points: np.ndarray,
        b_points: np.ndarray,
        *,
        diameter_a: float = 1.0,
        diameter_b: float = 1.0,
        seg_confidence: float = 0.8,
        at_boundary: bool = False,
        image_agreement: float = 0.5,
        model_evidence: float = 0.5,
    ) -> BranchVerdict:
        a = np.asarray(a_points, dtype=np.float64).reshape(-1, 3)
        b = np.asarray(b_points, dtype=np.float64).reshape(-1, 3)
        # endpoints = closest pair
        dmin = math.inf
        ea = a[0]
        eb = b[0]
        for p in a:
            for q in b:
                d = float(np.linalg.norm(p - q))
                if d < dmin:
                    dmin, ea, eb = d, p, q
        axis_a = _pca_axis(a)
        axis_b = _pca_axis(b)
        # direction from a toward b at endpoints
        link = eb - ea
        ln = np.linalg.norm(link)
        link_u = link / ln if ln > 1e-9 else axis_a
        dir_agree = abs(float(np.dot(axis_a, axis_b)))
        link_align = abs(float(np.dot(axis_a, link_u))) * abs(float(np.dot(axis_b, link_u)))
        diam_ratio = min(diameter_a, diameter_b) / max(max(diameter_a, diameter_b), 1e-6)
        morph = diam_ratio * (0.5 + 0.5 * dir_agree)

        ev: list[BranchEvidence] = []
        # geometric continuity
        if dmin <= 5.0:
            ev.append(BranchEvidence("geometric_continuity", "MERGE", min(1.0, 1.0 - dmin / 5.0),
                                     f"endpoints separated by {dmin:.1f} voxels"))
        else:
            ev.append(BranchEvidence("geometric_continuity", "SPLIT", min(1.0, (dmin - 5.0) / 20.0),
                                     f"endpoints separated by {dmin:.1f} voxels"))
        ev.append(BranchEvidence("direction_vectors", "MERGE" if dir_agree > 0.7 else "SPLIT",
                                 dir_agree if dir_agree > 0.7 else 1.0 - dir_agree,
                                 f"direction agreement {dir_agree:.2f}, link align {link_align:.2f}"))
        ev.append(BranchEvidence("diameter_consistency", "MERGE" if diam_ratio > 0.7 else "SPLIT",
                                 diam_ratio if diam_ratio > 0.7 else 1.0 - diam_ratio,
                                 f"similar local diameter ratio {diam_ratio:.2f}"))
        ev.append(BranchEvidence("segmentation_confidence", "NEUTRAL" if seg_confidence >= 0.6 else "SPLIT",
                                 1.0 - seg_confidence if seg_confidence < 0.6 else seg_confidence * 0.2,
                                 f"segmentation confidence {seg_confidence:.2f}"))
        if at_boundary:
            ev.append(BranchEvidence("boundary_location", "NEUTRAL", 0.3, "objects meet near chunk boundary"))
        ev.append(BranchEvidence("morphological_compatibility", "MERGE" if morph > 0.6 else "SPLIT",
                                 morph if morph > 0.6 else 1.0 - morph,
                                 f"morph score {morph:.2f}"))
        ev.append(BranchEvidence("local_imagery", "MERGE" if image_agreement > 0.55 else "SPLIT",
                                 abs(image_agreement - 0.5) * 2,
                                 f"image agreement {image_agreement:.2f}"))
        ev.append(BranchEvidence("model_evidence", "MERGE" if model_evidence > 0.55 else "SPLIT",
                                 abs(model_evidence - 0.5) * 2,
                                 f"model evidence {model_evidence:.2f}"))

        merge = sum(e.score for e in ev if e.supports == "MERGE")
        split = sum(e.score for e in ev if e.supports == "SPLIT")
        # normalize roughly
        tot = max(merge + split, 1e-6)
        merge_n, split_n = merge / tot, split / tot

        if abs(merge_n - split_n) < 0.15 or seg_confidence < 0.5:
            decision = "REVIEW"
        elif merge_n >= 0.62:
            decision = "MERGE"
        elif split_n >= 0.62:
            decision = "SPLIT"
        else:
            decision = "REVIEW"

        strongest = [e.detail for e in sorted(
            [e for e in ev if e.supports == ("MERGE" if merge_n >= split_n else "SPLIT")],
            key=lambda e: -e.score)[:3]]
        conflicting = [e.detail for e in ev if e.supports == ("SPLIT" if merge_n >= split_n else "MERGE") and e.score > 0.3]
        if seg_confidence < 0.6:
            conflicting.append(f"low segmentation confidence at junction ({seg_confidence:.2f})")

        region = {
            "centroid_a": a.mean(axis=0).tolist(),
            "centroid_b": b.mean(axis=0).tolist(),
            "endpoint_distance": dmin,
        }
        verdict = BranchVerdict(decision, merge_n, split_n, strongest, conflicting, ev, region)
        write_tool_receipt("branchjudge", verdict.as_dict())
        out = REPO / "receipts" / "branchjudge" / "latest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(verdict.as_dict(), indent=2) + chr(10), encoding="utf-8")
        return verdict
