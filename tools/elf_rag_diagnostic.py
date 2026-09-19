"""Persist RAG and edge-cost evidence for an immutable FRMC fragment volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import elf.segmentation.features as features
import elf.segmentation.multicut as multicut
import numpy as np
import tifffile


def summary(values: np.ndarray) -> dict[str, float]:
    return {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std()),
            "p01": float(np.percentile(values, 1)), "p50": float(np.percentile(values, 50)), "p99": float(np.percentile(values, 99))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--affinities", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    affinities = np.load(args.affinities, allow_pickle=False)
    boundaries = tifffile.imread(args.boundaries)
    fragments = np.load(args.fragments, allow_pickle=False)
    fused = np.minimum(affinities, boundaries[None, ...])
    split_affinities = 1.0 - fused
    watershed_input = np.maximum(split_affinities[1], split_affinities[2])
    rag = features.compute_rag(fragments)
    feature = features.compute_affinity_features(
        rag, fragments, split_affinities, [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    )[:, 0]
    edge_sizes = features.compute_boundary_mean_and_length(rag, fragments, watershed_input)[:, 1]
    transformed = multicut.transform_probabilities_to_costs(feature, edge_sizes=edge_sizes, beta=args.beta)
    ids, counts = np.unique(fragments, return_counts=True)
    z_extent = []
    for label in ids:
        locations = np.where(fragments == label)[0]
        z_extent.append(int(locations.max() - locations.min() + 1))
    result = {
        "schema_version": 1, "classification": "MACHINE_PSEUDOLABEL_DIAGNOSTIC",
        "fragment_count": int(ids.size), "fragment_voxel_count": summary(counts.astype(np.float64)),
        "fragment_z_extent": summary(np.asarray(z_extent, dtype=np.float64)),
        "rag": {"nodes": int(rag.numberOfNodes), "edges": int(rag.numberOfEdges)},
        "affinity_feature": summary(feature), "edge_sizes": summary(edge_sizes), "transformed_cost": summary(transformed),
        "transformed_cost_positive_fraction": float((transformed > 0).mean()),
        "transformed_cost_negative_fraction": float((transformed < 0).mean()),
        "finite": bool(np.isfinite(feature).all() and np.isfinite(edge_sizes).all() and np.isfinite(transformed).all()),
        "beta": args.beta,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
