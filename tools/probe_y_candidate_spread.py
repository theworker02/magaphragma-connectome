"""Probe how many distinct Y-oriented raw-EM interface centres a TRAIN crop has,
and their raw-gradient score spread. Label-blind: reads raw EM only.

This informs how deep to page for fresh Y candidates. It never reads or writes
any SAME/DIFFERENT decision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def probe(workspace_path: Path, top: int = 400) -> dict:
    w = json.loads(workspace_path.read_text())
    raw = np.load(w["raw"]["path"], mmap_mode="r", allow_pickle=False)
    arr = np.asarray(raw, dtype=np.float32)
    grads = np.stack(np.gradient(arr))
    score = np.linalg.norm(grads, axis=0)
    margin = 3
    order = np.argsort(score.ravel())[::-1]
    y_centres = []
    for flat in order[: top * 8]:
        z, y, x = map(int, np.unravel_index(flat, score.shape))
        if min(z, y, x) < margin or z >= arr.shape[0] - margin or y >= arr.shape[1] - margin or x >= arr.shape[2] - margin:
            continue
        axis = int(np.argmax(np.abs(grads[(slice(None),) + (z, y, x)])))
        if axis != 1:
            continue
        # spatial separation like the generator
        if all((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 >= 12 ** 2 for c in y_centres):
            y_centres.append((z, y, x, float(score[z, y, x])))
        if len(y_centres) >= top:
            break
    scores = [c[3] for c in y_centres]
    return {
        "crop_id": w["crop_id"],
        "distinct_Y_centres_found": len(y_centres),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_median": float(np.median(scores)) if scores else None,
    }


if __name__ == "__main__":
    results = []
    for crop in sys.argv[1:]:
        ws = REPO / "experiments/phase6e/g3-external-review-packages-003" / crop / "workspace.json"
        results.append(probe(ws))
    print(json.dumps(results, indent=2))
