"""Run the documented SegNeuron FRMC stages separately in its ELF environment.

Invoke with the isolated Linux/Micromamba environment.  The implementation
retains the source postprocessor's probability fusion, 2-D watershed settings,
and ELF RAG/multicut operations, but persists the fragment volume before
agglomeration for diagnosis.  Outputs are proposal-only, never MV-SEG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _probabilities(values, name: str, dimensions: int):
    import numpy as np
    array = np.asarray(values)
    if array.ndim != dimensions or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be a nonempty {dimensions}-D array")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return np.ascontiguousarray(array, dtype=np.float32)


def run_stages(affinity_path: Path, boundary_path: Path, fragments_output: Path, labels_output: Path, receipt_output: Path, beta: float) -> dict:
    import numpy as np
    import tifffile
    import elf.segmentation.features as feats
    import elf.segmentation.multicut as mc
    import elf.segmentation.watershed as ws

    if not 0 < beta < 1:
        raise ValueError("beta must be strictly between zero and one")
    if any(path.exists() for path in (fragments_output, labels_output, receipt_output)):
        raise FileExistsError("Stage outputs are immutable; choose a new experiment directory")
    affinities = _probabilities(np.load(affinity_path, allow_pickle=False), "affinities", 4)
    boundaries = _probabilities(tifffile.imread(boundary_path), "boundaries", 3)
    if affinities.shape[0] != 3 or affinities.shape[1:] != boundaries.shape:
        raise ValueError("Expected affinity CZYX and matching boundary ZYX volumes")

    # Exact fusion/direction used by the retained FRMC postprocessor.
    fused = np.minimum(affinities, boundaries[None])
    split_affinities = 1.0 - fused
    watershed_input = np.maximum(split_affinities[1], split_affinities[2])
    fragments = np.empty(watershed_input.shape, dtype=np.uint64)
    offset = 0
    for z in range(fragments.shape[0]):
        slice_labels, _ = ws.distance_transform_watershed(watershed_input[z], threshold=0.25, sigma_seeds=2.0)
        ids, inverse = np.unique(np.asarray(slice_labels), return_inverse=True)
        if ids.size == 0 or np.any(ids <= 0):
            raise RuntimeError(f"ELF fragment generation returned an invalid slice at z={z}")
        fragments[z] = inverse.reshape(slice_labels.shape).astype(np.uint64) + offset + 1
        offset += ids.size
    if offset > np.iinfo(np.uint32).max:
        raise OverflowError("Too many fragments for uint32 output")

    rag = feats.compute_rag(fragments)
    if rag.numberOfEdges == 0:
        node_labels = np.arange(rag.numberOfNodes, dtype=np.uint64)
    else:
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        costs = feats.compute_affinity_features(rag, fragments, split_affinities, offsets)[:, 0]
        edge_sizes = feats.compute_boundary_mean_and_length(rag, fragments, watershed_input)[:, 1]
        if not np.isfinite(costs).all() or not np.isfinite(edge_sizes).all() or np.any(edge_sizes <= 0):
            raise RuntimeError("ELF produced invalid RAG features")
        node_labels = mc.multicut_kernighan_lin(rag, mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes, beta=beta))
    labels = feats.project_node_labels_to_pixels(rag, fragments, node_labels)
    _, inverse = np.unique(labels, return_inverse=True)
    final = (inverse.reshape(fragments.shape) + 1).astype(np.uint32)
    fragments_u32 = fragments.astype(np.uint32)
    for path in (fragments_output, labels_output, receipt_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(fragments_output, fragments_u32, allow_pickle=False)
    np.save(labels_output, final, allow_pickle=False)
    result = {
        "kind": "SEGNEURON_FRMC_STAGE_EXPERIMENT_V1", "status": "MACHINE_PSEUDOLABEL", "review_state": "REVIEW_REQUIRED",
        "raw_network_prediction": {"affinities": str(affinity_path.resolve()), "boundaries": str(boundary_path.resolve())},
        "fragment_generation": {"algorithm": "ELF distance_transform_watershed", "threshold": 0.25, "sigma_seeds": 2.0, "fragments_path": str(fragments_output.resolve()), "fragment_count": int(fragments_u32.max())},
        "agglomeration": {"algorithm": "ELF RAG affinity features plus multicut_kernighan_lin", "beta": beta, "labels_path": str(labels_output.resolve()), "instance_count": int(final.max())},
        "shape_zyx": list(final.shape), "prohibited_uses": ["ground_truth", "held_out_metrics", "production_mv_seg", "biological_identity"],
    }
    receipt_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affinities", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--beta", type=float, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_stages(args.affinities, args.boundaries, args.fragments, args.labels, args.receipt, args.beta), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
