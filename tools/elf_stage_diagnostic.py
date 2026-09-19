"""Persist stage-local diagnostics for a preserved SegNeuron/ELF run.

This is deliberately observational: it neither changes predictions nor tunes
watershed or multicut parameters.  It records the exact input to the existing
ELF watershed call so later experiments can distinguish network output from
fragment-generation behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def describe(array: np.ndarray) -> dict[str, float | int | list[int]]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "finite_fraction": float(np.isfinite(array).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    affinity_path = args.prediction_dir / "affinities.npy"
    boundary_path = args.prediction_dir / "boundaries.tif"
    affinities = np.load(affinity_path)
    boundaries = tifffile.imread(boundary_path)
    if affinities.shape[0] != 3:
        raise ValueError(f"Expected three affinity channels, found {affinities.shape}")
    if boundaries.shape != affinities.shape[1:]:
        raise ValueError(f"Boundary/affinity mismatch: {boundaries.shape} vs {affinities.shape}")

    # This exactly matches segneuron_frmc_stages.py.  Affinity channel labels
    # are intentionally recorded as upstream assumptions, not asserted biology.
    fused = np.minimum(affinities, boundaries[None, ...])
    split_signal = 1.0 - fused
    watershed_input = np.maximum(split_signal[1], split_signal[2])
    slices = []
    for z_index, image in enumerate(watershed_input):
        slices.append(
            {
                "z_index": z_index,
                "min": float(image.min()),
                "max": float(image.max()),
                "mean": float(image.mean()),
                "std": float(image.std()),
                "constant_input": bool(float(image.max()) == float(image.min())),
                "finite_fraction": float(np.isfinite(image).mean()),
            }
        )
    payload = {
        "schema_version": 1,
        "classification": "MACHINE_PSEUDOLABEL_DIAGNOSTIC",
        "immutable_input": {
            "affinities": str(affinity_path),
            "boundaries": str(boundary_path),
        },
        "operation": {
            "fused": "minimum(affinities, boundaries[None, ...])",
            "split_signal": "1.0 - fused",
            "watershed_input": "maximum(split_signal[1], split_signal[2])",
            "upstream_watershed_call": "elf.segmentation.watershed.distance_transform_watershed(slice, threshold=0.25, sigma_seeds=2)",
            "upstream_divide_warning_site": "elf/segmentation/watershed.py:163: dt = 1 - (dt - dt.min()) / dt.max()",
            "warning_interpretation": "A divide warning is an upstream distance-transform condition; this file records its exact pre-call slice inputs and does not claim causation without instrumented execution.",
        },
        "affinities": [describe(affinities[index]) for index in range(3)],
        "boundaries": describe(boundaries),
        "fused": [describe(fused[index]) for index in range(3)],
        "watershed_input": describe(watershed_input),
        "watershed_slices": slices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
