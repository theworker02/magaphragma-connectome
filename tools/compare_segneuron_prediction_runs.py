"""Compare immutable raw SegNeuron predictions before instance processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def statistics(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "percentiles": {str(percentile): float(np.percentile(values, percentile)) for percentile in (1, 5, 50, 95, 99)},
        "fraction_ge_0_5": float((values >= 0.5).mean()),
        "fraction_ge_0_9": float((values >= 0.9).mean()),
        "mean_absolute_neighbor_delta": {
            "z": float(np.abs(np.diff(values, axis=0)).mean()),
            "y": float(np.abs(np.diff(values, axis=1)).mean()),
            "x": float(np.abs(np.diff(values, axis=2)).mean()),
        },
    }


def describe(directory: Path) -> dict[str, object]:
    affinities = np.load(directory / "affinities.npy", allow_pickle=False)
    foreground = tifffile.imread(directory / "boundaries.tif")
    fused = np.minimum(affinities, foreground[None, ...])
    watershed_input = np.maximum(1.0 - fused[1], 1.0 - fused[2])
    threshold = 0.25
    slice_coverage = (watershed_input > threshold).mean(axis=(1, 2))
    return {
        "directory": str(directory),
        "affinities": [statistics(affinities[channel]) for channel in range(3)],
        "foreground": statistics(foreground),
        "mean_affinity": statistics(affinities.mean(axis=0)),
        "watershed_gate": {
            "threshold": threshold,
            "input": statistics(watershed_input),
            "voxel_fraction_above_threshold": float((watershed_input > threshold).mean()),
            "all_foreground_slices": int((slice_coverage == 1.0).sum()),
            "no_foreground_slices": int((slice_coverage == 0.0).sum()),
            "slice_coverage_min": float(slice_coverage.min()),
            "slice_coverage_max": float(slice_coverage.max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-shot", type=Path, required=True)
    parser.add_argument("--step50", type=Path, required=True)
    parser.add_argument("--step250", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = {
        "schema_version": 1,
        "classification": "MACHINE_PSEUDOLABEL_DIAGNOSTIC_ONLY",
        "runs": {"zero_shot": describe(args.zero_shot), "step50": describe(args.step50), "step250": describe(args.step250)},
        "prohibited_uses": ["ground_truth", "production_mv_seg", "biological_identity"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
