"""Empirically document ELF multicut cost/sign behavior on synthetic graph fixtures.

The fixtures exercise only a two-node graph.  They are algorithm diagnostics,
never biological data or segmentation results.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import elf.segmentation.multicut as multicut
import nifty.graph
import numpy as np


def solve(split_probability: float) -> dict[str, object]:
    graph = nifty.graph.undirectedGraph(2)
    graph.insertEdges(np.array([[0, 1]], dtype="uint64"))
    cost = multicut.transform_probabilities_to_costs(
        np.array([split_probability], dtype="float64"), edge_sizes=np.array([1.0]), beta=0.1
    )
    labels = multicut.multicut_kernighan_lin(graph, cost)
    return {
        "split_probability_input": split_probability,
        "transformed_cost": float(cost[0]),
        "node_labels": [int(value) for value in labels],
        "solver_action": "MERGE" if labels[0] == labels[1] else "CUT",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New immutable synthetic-fixture receipt")
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite synthetic fixture receipt: {output}")
    result = {
        "schema_version": 1,
        "classification": "SYNTHETIC_ALGORITHM_DIAGNOSTIC_NOT_BIOLOGICAL_DATA",
        "graph": {"nodes": 2, "edges": [[0, 1]]},
        "transform": "elf.segmentation.multicut.transform_probabilities_to_costs(probability, edge_sizes=[1], beta=0.1)",
        "fixtures": {
            "merge_expected_low_split_probability": solve(0.01),
            "cut_expected_high_split_probability": solve(0.99),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
