"""Call BranchJudge on SeamSmith REVIEW candidates."""
from __future__ import annotations

from typing import Any

import numpy as np


def assist_review_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from branchjudge.judge import BranchJudge

    bj = BranchJudge()
    out: list[dict[str, Any]] = []
    for c in candidates:
        if c.get("recommended_action") != "REVIEW":
            continue
        ev = c.get("evidence") or {}
        dist = float(ev.get("spatial_distance", 3.0))
        a_pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]], dtype=float)
        b_pts = np.array([[0.0, 0.0, 5.0 + dist], [0.2, 0.0, 10.0 + dist]], dtype=float)
        verdict = bj.judge(
            a_pts, b_pts,
            diameter_a=2.0, diameter_b=2.0,
            seg_confidence=float(ev.get("confidence", 0.45)),
            at_boundary=True,
            image_agreement=float(ev.get("morphology_agreement", 0.5)),
            model_evidence=float(ev.get("direction_agreement", 0.5)),
        )
        entry = {
            "a_label": (c.get("a") or {}).get("label_id"),
            "b_label": (c.get("b") or {}).get("label_id"),
            "seam_action": "REVIEW",
            "branchjudge": verdict.as_dict(),
        }
        c["branchjudge"] = {
            "decision": verdict.decision,
            "merge_evidence": verdict.merge_evidence,
            "split_evidence": verdict.split_evidence,
        }
        out.append(entry)
    return out
