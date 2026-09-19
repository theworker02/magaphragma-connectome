"""Append an immutable DVID target-regression summary to the campaign table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage


def stats(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "std": float(values.std()), "p01": float(np.percentile(values, 1)), "p50": float(np.percentile(values, 50)), "p99": float(np.percentile(values, 99)), "fraction_ge_0_9": float((values >= 0.9).mean())}


def component_diagnostics(labels: np.ndarray) -> dict[str, int]:
    ids = np.unique(labels)
    disconnected_instances = 0
    total_components = 0
    structure = ndimage.generate_binary_structure(3, 1)
    for value in ids:
        count = int(ndimage.label(labels == value, structure=structure)[1])
        total_components += count
        disconnected_instances += int(count > 1)
    return {"label_instances": int(ids.size), "connected_components_across_labels": total_components, "disconnected_label_instances": disconnected_instances}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--true3d-dir", required=True, type=Path)
    parser.add_argument("--edge-receipt", required=True, type=Path)
    parser.add_argument("--edge-table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    affinity = np.load(args.prediction_dir / "affinities.npy", allow_pickle=False)
    foreground = tifffile.imread(args.prediction_dir / "boundaries.tif")
    stage = json.loads((args.true3d_dir / "receipt.json").read_text(encoding="utf-8"))
    edge = json.loads(args.edge_receipt.read_text(encoding="utf-8"))
    edge_table = np.load(args.edge_table, allow_pickle=False)
    fragments = np.load(args.true3d_dir / "fragments.npy", allow_pickle=False)
    labels = np.load(args.true3d_dir / "labels.npy", allow_pickle=False)
    _, fragment_sizes = np.unique(fragments, return_counts=True)
    _, instance_sizes = np.unique(labels, return_counts=True)
    split = edge["split_probability"]
    cost = edge["cost_sign"]
    row = {
        "step": args.step, "classification": "MACHINE_PSEUDOLABEL_DIAGNOSTIC",
        "affinity": stats(affinity), "foreground": stats(foreground),
        "watershed_fragments": stage["fragments"], "rag": {"nodes": stage["rag"]["nodes"], "edges": stage["rag"]["edges"]},
        "split_probability": {**split, "edges_gt_0_5": int((edge_table["split_probability"] > 0.5).sum()),
                              "edges_gt_0_75": int((edge_table["split_probability"] > 0.75).sum()), "edges_gt_0_9": int((edge_table["split_probability"] > 0.9).sum())},
        "multicut": {"instance_count": int(instance_sizes.size), "largest_instance_fraction": float(instance_sizes.max() / labels.size), "positive_cost_count": int(round(cost["positive_fraction"] * edge["edge_count"])), "negative_cost_count": int(round(cost["negative_fraction"] * edge["edge_count"]))},
        "qa": {"tiny_watershed_fragment_fraction_lt_512_voxels": float((fragment_sizes < 512).mean()), "disconnected_components": component_diagnostics(labels), "result": "FAIL_SINGLE_INSTANCE" if instance_sizes.size == 1 else "REVIEW_REQUIRED"},
        "artifacts": {"prediction_dir": str(args.prediction_dir), "true3d_dir": str(args.true3d_dir), "edge_receipt": str(args.edge_receipt), "edge_table": str(args.edge_table)},
    }
    document = {"schema_version": 1, "progression": []}
    if args.output.exists():
        document = json.loads(args.output.read_text(encoding="utf-8"))
    if any(item["step"] == args.step for item in document["progression"]):
        raise ValueError(f"Step {args.step} already exists in progression table")
    document["progression"].append(row)
    document["progression"].sort(key=lambda item: item["step"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
