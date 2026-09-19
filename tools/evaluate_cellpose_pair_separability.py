"""Measure Cellpose raw-output separation at immutable reviewed G3 pairs.

Cellpose emits flows and a cell-probability field, not neuronal affinities.
This tool therefore makes no affinity or biological claim. It records whether
fixed, local output-derived descriptors differ between already-reviewed SAME
and DIFFERENT pairs in a frozen G3 region. Any later held-out test must use
the descriptor orientation fixed here without re-fitting on that held-out data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(values: list[float]) -> dict[str, object]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "quantiles_p10_p50_p90": [float(value) for value in np.quantile(data, [0.1, 0.5, 0.9])],
    }


def auc_without_orientation(values: list[float], labels: list[int]) -> dict[str, float]:
    """Return rank AUC in both orientations; this avoids a post-hoc sign claim.

    A class label of one denotes expert-reviewed DIFFERENT_PROCESS. The raw
    descriptor itself has no asserted Cellpose boundary semantics, so both
    score directions are retained rather than choosing the nicer one.
    """
    scores, truth = np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=np.int8)
    positive, negative = scores[truth == 1], scores[truth == 0]
    if not positive.size or not negative.size:
        raise ValueError("Both reviewed SAME and DIFFERENT evidence are required")
    greater = sum(float(left > right) + 0.5 * float(left == right) for left in positive for right in negative)
    auc = greater / float(positive.size * negative.size)
    return {"different_high_auc": float(auc), "different_low_auc": float(1.0 - auc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-receipt", type=Path, required=True)
    parser.add_argument("--flows", type=Path, required=True)
    parser.add_argument("--cell-probability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("Refusing to overwrite immutable Cellpose separability evidence")

    receipt = json.loads(arguments.region_receipt.read_text(encoding="utf-8"))
    flows = np.load(arguments.flows, allow_pickle=False)
    probability = np.load(arguments.cell_probability, allow_pickle=False)
    if flows.shape != (3,) + probability.shape or not np.isfinite(flows).all() or not np.isfinite(probability).all():
        raise ValueError("Expected finite Cellpose CZYX flows and aligned ZYX cell probability")

    rows: list[dict[str, object]] = []
    for pair in receipt["eligible_pairs"]:
        channel, left = int(pair["channel_zyx"]), tuple(pair["pair_left_zyx"])
        right = tuple(pair["pair_right_zyx"])
        left_flow, right_flow = flows[(slice(None),) + left], flows[(slice(None),) + right]
        left_norm, right_norm = float(np.linalg.norm(left_flow)), float(np.linalg.norm(right_flow))
        cosine = float(np.dot(left_flow, right_flow) / max(left_norm * right_norm, 1e-12))
        rows.append({
            "interface_id": pair["interface_id"],
            "decision": pair["decision"],
            "channel_zyx": channel,
            "left_zyx": list(left),
            "right_zyx": list(right),
            "loss_weight": pair["loss_weight"],
            "cell_probability_abs_delta": float(abs(probability[left] - probability[right])),
            "flow_l2_delta": float(np.linalg.norm(left_flow - right_flow)),
            "flow_cosine_similarity": cosine,
        })
    features = ("cell_probability_abs_delta", "flow_l2_delta", "flow_cosine_similarity")
    labels = [1 if row["decision"] == "DIFFERENT_PROCESS" else 0 for row in rows]
    report = {
        "status": "EXPLORATORY_RAW_OUTPUT_PAIR_SEPARABILITY_ONLY",
        "region": receipt["crop_id"],
        "region_receipt": {"path": str(arguments.region_receipt.resolve()), "sha256": sha256(arguments.region_receipt)},
        "flows": {"path": str(arguments.flows.resolve()), "sha256": sha256(arguments.flows)},
        "cell_probability": {"path": str(arguments.cell_probability.resolve()), "sha256": sha256(arguments.cell_probability)},
        "features": {
            feature: {
                "same": summary([float(row[feature]) for row in rows if row["decision"] == "SAME_PROCESS"]),
                "different": summary([float(row[feature]) for row in rows if row["decision"] == "DIFFERENT_PROCESS"]),
                "auc_without_posthoc_orientation": auc_without_orientation([float(row[feature]) for row in rows], labels),
            }
            for feature in features
        },
        "pairs": rows,
        "scientific_boundary": "These are Cellpose output descriptors, not documented neuronal-affinity channels or independently validated membrane probabilities. They cannot by themselves enter training or establish a biological boundary.",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
