"""Persist every true-3-D RAG edge's prediction-to-cost evidence chain.

This is a diagnostic over frozen machine predictions.  It cannot create an
instance, biological record, or change a postprocessing parameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import elf.segmentation.features as features
import elf.segmentation.multicut as multicut
import numpy as np
import tifffile

OFFSETS = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))


def axes_for_offset(shape: tuple[int, int, int], offset: tuple[int, int, int]) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    source: list[slice] = []
    target: list[slice] = []
    for size, delta in zip(shape, offset):
        if delta < 0:
            source.append(slice(-delta, size))
            target.append(slice(0, size + delta))
        elif delta > 0:
            source.append(slice(0, size - delta))
            target.append(slice(delta, size))
        else:
            source.append(slice(0, size))
            target.append(slice(0, size))
    return tuple(source), tuple(target)


def aggregates(edge_indices: np.ndarray, values: np.ndarray, edge_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(edge_indices, minlength=edge_count).astype("uint64")
    sums = np.bincount(edge_indices, weights=values, minlength=edge_count)
    means = np.divide(sums, counts, out=np.full(edge_count, np.nan), where=counts > 0)
    minimum = np.full(edge_count, np.inf)
    maximum = np.full(edge_count, -np.inf)
    np.minimum.at(minimum, edge_indices, values)
    np.maximum.at(maximum, edge_indices, values)
    minimum[counts == 0] = np.nan
    maximum[counts == 0] = np.nan
    return counts, means, minimum, maximum


def edge_quantiles(edge_indices: np.ndarray, values: np.ndarray, edge_count: int) -> np.ndarray:
    """Return p01/p05/p50/p95/p99 for every populated edge interface."""
    result = np.full((edge_count, 5), np.nan, dtype="float32")
    order = np.argsort(edge_indices, kind="stable")
    sorted_edges = edge_indices[order]
    sorted_values = values[order]
    ids, starts = np.unique(sorted_edges, return_index=True)
    ends = np.r_[starts[1:], len(sorted_edges)]
    for edge_id, start, end in zip(ids, starts, ends):
        result[edge_id] = np.percentile(sorted_values[start:end], [1, 5, 50, 95, 99])
    return result


def numeric_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    return {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std()),
            "p01": float(np.percentile(values, 1)), "p50": float(np.percentile(values, 50)), "p99": float(np.percentile(values, 99))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--affinities", type=Path, required=True)
    parser.add_argument("--foreground", type=Path, required=True)
    parser.add_argument("--fragments", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    affinity = np.load(args.affinities, allow_pickle=False).astype("float32", copy=False)
    foreground = tifffile.imread(args.foreground).astype("float32", copy=False)
    fragments = np.load(args.fragments, allow_pickle=False)
    if affinity.shape != (3,) + foreground.shape or fragments.shape != foreground.shape:
        raise ValueError("CZYX affinity, ZYX foreground, and fragment shapes must agree")
    fused = np.minimum(affinity, foreground[None, ...])
    split = 1.0 - fused
    rag = features.compute_rag(fragments)
    uv_ids = np.asarray(rag.uvIds(), dtype="uint64")
    standard_features = features.compute_affinity_features(rag, fragments, split, list(OFFSETS))
    split_probability = standard_features[:, 0]
    watershed_input = np.maximum(split[1], split[2])
    edge_sizes = features.compute_boundary_mean_and_length(rag, fragments, watershed_input)[:, 1]
    unweighted_costs = multicut.transform_probabilities_to_costs(split_probability, beta=args.beta)
    costs = multicut.transform_probabilities_to_costs(split_probability, edge_sizes=edge_sizes, beta=args.beta)
    node_labels = multicut.multicut_kernighan_lin(rag, costs)
    actual_solver_action = np.where(node_labels[uv_ids[:, 0]] == node_labels[uv_ids[:, 1]], 0, 1).astype("uint8")
    expected_action = np.where(costs > 0, 0, np.where(costs < 0, 1, 2)).astype("uint8")
    edge_count = len(uv_ids)
    edge_lookup = {tuple(sorted((int(left), int(right)))): index for index, (left, right) in enumerate(uv_ids)}
    shape = tuple(int(value) for value in fragments.shape)
    evidence = {name: np.full((edge_count, 3, 3), np.nan, dtype="float32") for name in ("affinity", "foreground", "fused_merge", "split")}
    affinity_quantiles = np.full((edge_count, 3, 5), np.nan, dtype="float32")
    interface_counts = np.zeros((edge_count, 3), dtype="uint64")
    unmatched_pairs = []
    for channel, offset in enumerate(OFFSETS):
        source, target = axes_for_offset(shape, offset)
        source_labels = fragments[source].ravel()
        target_labels = fragments[target].ravel()
        crossing = source_labels != target_labels
        left = source_labels[crossing]
        right = target_labels[crossing]
        ordered = np.stack((np.minimum(left, right), np.maximum(left, right)), axis=1)
        pairs, inverse = np.unique(ordered, axis=0, return_inverse=True)
        local_to_global = np.empty(len(pairs), dtype="int64")
        for pair_index, pair in enumerate(pairs):
            edge_id = edge_lookup.get((int(pair[0]), int(pair[1])))
            if edge_id is None:
                unmatched_pairs.append([int(pair[0]), int(pair[1]), channel])
                local_to_global[pair_index] = -1
            else:
                local_to_global[pair_index] = edge_id
        keep = local_to_global[inverse] >= 0
        edge_indices = local_to_global[inverse][keep]
        source_flat = np.flatnonzero(crossing)
        raw_values = affinity[channel][source].ravel()[source_flat][keep]
        foreground_values = foreground[source].ravel()[source_flat][keep]
        fused_values = fused[channel][source].ravel()[source_flat][keep]
        split_values = split[channel][source].ravel()[source_flat][keep]
        for name, values in (("affinity", raw_values), ("foreground", foreground_values), ("fused_merge", fused_values), ("split", split_values)):
            count, mean, minimum, maximum = aggregates(edge_indices, values, edge_count)
            if name == "affinity":
                interface_counts[:, channel] = count
            evidence[name][:, channel, :] = np.stack((mean, minimum, maximum), axis=1)
        affinity_quantiles[:, channel, :] = edge_quantiles(edge_indices, raw_values, edge_count)
    aggregation_error = {
        "matches_interface_split_mean": float(np.nanmax(np.abs(split_probability - np.nanmean(evidence["split"][:, :, 0], axis=1)))),
        "matches_interface_split_min": float(np.nanmax(np.abs(split_probability - np.nanmin(evidence["split"][:, :, 1], axis=1)))),
        "matches_interface_split_max": float(np.nanmax(np.abs(split_probability - np.nanmax(evidence["split"][:, :, 2], axis=1)))),
    }
    pooled_counts = interface_counts.sum(axis=1)
    pooled_split_mean = np.divide((evidence["split"][:, :, 0] * interface_counts).sum(axis=1), pooled_counts,
                                  out=np.full(edge_count, np.nan), where=pooled_counts > 0)
    aggregation_error["matches_pooled_interface_split_mean"] = float(np.nanmax(np.abs(split_probability - pooled_split_mean)))
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, edge_id=np.arange(edge_count, dtype="uint64"), fragment_uv=uv_ids,
                        standard_features=standard_features, split_probability=split_probability,
                        cost_before_edge_weighting=unweighted_costs, transformed_cost=costs,
                        expected_action=expected_action, actual_solver_action=actual_solver_action,
                        interface_counts=interface_counts, affinity_quantiles=affinity_quantiles, **evidence)
    receipt = {
        "schema_version": 1, "classification": "MACHINE_PSEUDOLABEL_DIAGNOSTIC", "edge_count": edge_count,
        "offsets_zyx": [list(offset) for offset in OFFSETS],
        "channel_to_physical_xyz": [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
        "edge_table_schema": {"edge_id": "row index", "fragment_uv": "fragment IDs U,V", "interface_counts": "per Z/Y/X channel", "affinity": "per channel columns mean,min,max", "affinity_quantiles": "per channel p01,p05,p50,p95,p99", "foreground": "per channel mean,min,max", "fused_merge": "per channel mean,min,max after min(raw_affinity, foreground)", "split": "per channel mean,min,max after 1-fused_merge", "standard_features": "all ELF feature columns", "split_probability": "ELF standard feature column 0", "cost_before_edge_weighting": "negative-log-likelihood + beta bias", "transformed_cost": "edge-size-weighted final multicut cost", "expected_action": "0=MERGE,1=CUT,2=ZERO_COST", "actual_solver_action": "0=MERGE,1=CUT from the solver node labels"},
        "interface_aggregation": "independent source-voxel interface mean/min/max and affinity quantiles, retained per CZYX channel; exact ELF standard features retained without assuming unverified column semantics",
        "elf_feature": {"function": "compute_affinity_features(rag, split, offsets)", "selected_column": 0, "all_columns": int(standard_features.shape[1]), "verified_selected_column_semantics": "pooled interface-voxel-weighted mean of the three-channel split affinity", "selected_column_max_abs_error": aggregation_error["matches_pooled_interface_split_mean"]},
        "split_probability": numeric_summary(split_probability), "edge_sizes": numeric_summary(edge_sizes), "cost_before_edge_weighting": numeric_summary(unweighted_costs), "transformed_cost": numeric_summary(costs),
        "cost_sign": {"positive_fraction": float((costs > 0).mean()), "negative_fraction": float((costs < 0).mean()), "zero_fraction": float((costs == 0).mean())},
        "cost_transform_cut_threshold": {"beta": args.beta, "split_probability_for_zero_unweighted_cost": float(1.0 - args.beta), "derivation": "log((1-p)/p) + log((1-beta)/beta) = 0"},
        "actions": {"expected_merge_count": int((expected_action == 0).sum()), "expected_cut_count": int((expected_action == 1).sum()), "actual_merge_count": int((actual_solver_action == 0).sum()), "actual_cut_count": int((actual_solver_action == 1).sum())},
        "interface_counts": numeric_summary(interface_counts[interface_counts > 0]),
        "aggregation_comparison_max_abs_error": aggregation_error,
        "unmatched_fragment_adjacency_pairs": unmatched_pairs[:20], "unmatched_pair_count": len(unmatched_pairs),
        "artifacts": {"edge_table": str(args.output_npz), "affinities": str(args.affinities), "foreground": str(args.foreground), "fragments": str(args.fragments)},
        "prohibited_uses": ["ground_truth", "production_mv_seg", "biological_identity"],
    }
    args.output_json.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
