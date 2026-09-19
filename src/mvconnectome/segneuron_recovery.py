"""Stage-preserving diagnostics for SegNeuron proposal recovery experiments.

All outputs of this module are machine pseudolabel diagnostics.  It deliberately
has no path to ground-truth registration, metric qualification, or MV-SEG.
"""
from __future__ import annotations

import json
from pathlib import Path

from .io import sha256_file, write_json_atomic


def analyze_raw_prediction(inference_dir: Path, output: Path) -> dict:
    """Record immutable raw prediction statistics before any instance stage."""
    import numpy as np
    import tifffile

    receipt_path = inference_dir / "inference.json"
    affinities_path = inference_dir / "affinities.npy"
    boundaries_path = inference_dir / "boundaries.tif"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    affinities = np.load(affinities_path, allow_pickle=False)
    boundaries = tifffile.imread(boundaries_path)
    if affinities.ndim != 4 or affinities.shape[0] != 3:
        raise ValueError("Expected CZYX three-channel affinities")
    if boundaries.ndim != 3 or tuple(affinities.shape[1:]) != tuple(boundaries.shape):
        raise ValueError("Boundaries must be ZYX and match affinity spatial shape")
    if not (np.isfinite(affinities).all() and np.isfinite(boundaries).all()):
        raise ValueError("Raw prediction contains non-finite values")
    if affinities.min() < 0 or affinities.max() > 1 or boundaries.min() < 0 or boundaries.max() > 1:
        raise ValueError("Raw prediction is not a probability volume")

    def stats(values: object) -> dict:
        array = np.asarray(values)
        return {
            "min": float(array.min()), "max": float(array.max()), "mean": float(array.mean()),
            "fraction_ge_0_5": float((array >= 0.5).mean()),
            "fraction_ge_0_9": float((array >= 0.9).mean()),
        }

    channels = [stats(affinities[channel]) for channel in range(3)]
    mean_affinity = affinities.mean(axis=0)
    result = {
        "kind": "SEGNEURON_RAW_PREDICTION_DIAGNOSTIC_V1",
        "classification": "MACHINE_PSEUDOLABEL_DIAGNOSTIC_ONLY",
        "inference_receipt": str(receipt_path.resolve()),
        "inference_receipt_sha256": sha256_file(receipt_path),
        "raw_input_sha256": receipt["input"]["sha256"],
        "affinities": {"path": str(affinities_path.resolve()), "sha256": sha256_file(affinities_path), "axes": "CZYX", "channels": channels},
        "boundaries": {"path": str(boundaries_path.resolve()), "sha256": sha256_file(boundaries_path), "axes": "ZYX", "stats": stats(boundaries)},
        "mean_affinity": stats(mean_affinity),
        "collapse_signals": {
            "network_output_near_saturated": bool(mean_affinity.mean() >= 0.98),
            "network_output_near_universal_at_0_5": bool((mean_affinity >= 0.5).mean() >= 0.98),
            "interpretation": "RAW_PREDICTION_SATURATION_REQUIRES_DOMAIN_ADAPTATION_OR_INPUT_DIAGNOSIS" if mean_affinity.mean() >= 0.98 else "RAW_PREDICTION_NOT_SATURATED_BY_THIS_TEST",
        },
        "prohibited_uses": ["ground_truth", "held_out_metrics", "production_mv_seg", "biological_identity"],
    }
    write_json_atomic(output, result)
    return result
