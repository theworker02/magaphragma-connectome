"""Identify the exact preserved watershed-input slices that emit ELF warnings."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import tifffile
from elf.segmentation.watershed import distance_transform_watershed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    affinities = np.load(args.prediction_dir / "affinities.npy")
    boundaries = tifffile.imread(args.prediction_dir / "boundaries.tif")
    fused = np.minimum(affinities, boundaries[None, ...])
    watershed_input = np.maximum(1.0 - fused[1], 1.0 - fused[2])
    records = []
    for z_index, image in enumerate(watershed_input):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fragments, _ = distance_transform_watershed(image, threshold=0.25, sigma_seeds=2)
        records.append({
            "z_index": z_index,
            "threshold": 0.25,
            "thresholded_foreground_fraction": float((image > 0.25).mean()),
            "thresholded_all_foreground": bool(np.all(image > 0.25)),
            "input_min": float(image.min()),
            "input_max": float(image.max()),
            "input_std": float(image.std()),
            "fragment_labels": int(np.unique(fragments).size),
            "warnings": [
                {"category": item.category.__name__, "message": str(item.message), "filename": item.filename, "lineno": item.lineno}
                for item in caught
            ],
        })
    payload = {
        "schema_version": 1,
        "classification": "MACHINE_PSEUDOLABEL_DIAGNOSTIC",
        "operation": "distance_transform_watershed(slice, threshold=0.25, sigma_seeds=2)",
        "warning_slices": [record for record in records if record["warnings"]],
        "all_slice_records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
