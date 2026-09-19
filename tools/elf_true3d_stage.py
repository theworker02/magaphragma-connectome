"""Run a separate true-3-D watershed/RAG/multicut diagnostic.

This intentionally does not replace the upstream slice-wise baseline. It
reuses its retained fusion and affinity-cost interpretation solely to establish
whether the watershed dimensionality is the structural limitation.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import elf.segmentation.features as features
import elf.segmentation.multicut as multicut
import elf.segmentation.watershed as watershed
import numpy as np
import tifffile


def summary(values: np.ndarray) -> dict[str, float]:
    return {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std()),
            "p01": float(np.percentile(values, 1)), "p50": float(np.percentile(values, 50)), "p99": float(np.percentile(values, 99))}


def z_extent(labels: np.ndarray) -> np.ndarray:
    result = []
    for label in np.unique(labels):
        positions = np.where(labels == label)[0]
        result.append(int(positions.max() - positions.min() + 1))
    return np.asarray(result, dtype="float64")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--affinities", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Exclusive output directory required: {args.output_dir}")
    affinities = np.load(args.affinities, allow_pickle=False)
    boundaries = tifffile.imread(args.boundaries)
    if affinities.shape != (3,) + boundaries.shape:
        raise ValueError("Expected CZYX three-channel affinities with matching ZYX boundary volume")
    fused = np.minimum(affinities, boundaries[None, ...])
    split_affinities = 1.0 - fused
    watershed_input = np.maximum(split_affinities[1], split_affinities[2])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fragments, _ = watershed.distance_transform_watershed(
            watershed_input, threshold=0.25, sigma_seeds=2.0, pixel_pitch=(1.0, 1.0, 1.0)
        )
    fragments = np.asarray(fragments, dtype="uint32")
    rag = features.compute_rag(fragments)
    affinity_feature = features.compute_affinity_features(
        rag, fragments, split_affinities, [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    )[:, 0]
    edge_sizes = features.compute_boundary_mean_and_length(rag, fragments, watershed_input)[:, 1]
    costs = multicut.transform_probabilities_to_costs(affinity_feature, edge_sizes=edge_sizes, beta=args.beta)
    node_labels = multicut.multicut_kernighan_lin(rag, costs)
    labels = features.project_node_labels_to_pixels(rag, fragments, node_labels)
    _, inverse = np.unique(labels, return_inverse=True)
    labels = (inverse.reshape(fragments.shape) + 1).astype("uint32")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "fragments.npy", fragments, allow_pickle=False)
    np.save(args.output_dir / "labels.npy", labels, allow_pickle=False)
    result = {
        "schema_version": 1, "classification": "MACHINE_PSEUDOLABEL_DIAGNOSTIC", "review_state": "REVIEW_REQUIRED",
        "baseline_relationship": "SEPARATE_TRUE_3D_EXPERIMENT; upstream slice-wise baseline remains unchanged",
        "watershed": {"algorithm": "ELF distance_transform_watershed", "dimensions": "3D", "threshold": 0.25, "sigma_seeds": 2.0, "pixel_pitch_zyx": [1.0, 1.0, 1.0], "warnings": [str(item.message) for item in caught]},
        "fragments": {"count": int(np.unique(fragments).size), "z_extent": summary(z_extent(fragments))},
        "rag": {"nodes": int(rag.numberOfNodes), "edges": int(rag.numberOfEdges), "affinity_feature": summary(affinity_feature), "edge_sizes": summary(edge_sizes), "costs": summary(costs), "positive_cost_fraction": float((costs > 0).mean()), "negative_cost_fraction": float((costs < 0).mean())},
        "instances": {"count": int(np.unique(labels).size), "z_extent": summary(z_extent(labels))},
        "prohibited_uses": ["ground_truth", "production_mv_seg", "biological_identity"],
    }
    (args.output_dir / "receipt.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
