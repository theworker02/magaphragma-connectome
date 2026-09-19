"""Run one pinned Cellpose model on an eligible DVID crop without making masks.

This is the first, deliberately limited stage of the representation bake-off.
It records the model's direct 3-D flow and cell-probability outputs before
Cellpose dynamics/thresholding could turn them into candidate instances.
Those outputs are machine diagnostics only; they are not neuronal affinities,
ground truth, MV-FRAG records, or input to the protected 000004 volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CELLPOSE_SOURCE = ROOT / "third_party" / "cellpose"
DINOV3_SOURCE = ROOT / "third_party" / "dinov3"
CELLPOSE_OVERLAY = ROOT / "third_party" / "cellpose-runtime" / "site-packages"
CELLPOSE_MODELS = ROOT / "third_party" / "cellpose-runtime" / "models"


def sha256(path: Path) -> str:
    """Hash artifacts so a run can be reproduced or independently audited."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def statistics(array: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(array)
    values = array[finite]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_count": int(values.size),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles_p01_p05_p50_p95_p99": [float(item) for item in np.quantile(values, [0.01, 0.05, 0.50, 0.95, 0.99])],
    }


def import_cellpose(model_name: str) -> object:
    """Import the pinned source through an isolated dependency overlay.

    The available Windows torchvision wheel lacks the custom NMS operator for
    the qualified ROCm Torch build. Cellpose needs torchvision image transforms,
    not NMS, so declaring its schema is a minimal import compatibility adapter.
    No operator implementation is supplied or used; this adapter is recorded in
    every receipt and does not change Cellpose source or checkpoint weights.
    """
    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(CELLPOSE_MODELS)
    library = torch.library.Library("torchvision", "DEF")
    library.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    # CPDINO imports DINOv3 from the independently pinned official source. Its
    # Cellpose checkpoint supplies the encoder state; this does not fetch a
    # second DINO checkpoint or borrow any SegNeuron-derived representation.
    if model_name.startswith("cpdino"):
        if not DINOV3_SOURCE.is_dir():
            raise FileNotFoundError("Pinned DINOv3 source is required for CellposeDINO")
        sys.path.insert(0, str(DINOV3_SOURCE))
    sys.path.insert(0, str(CELLPOSE_SOURCE))
    # Add the overlay after the qualified environment: its newer NumPy must not
    # shadow the ROCm environment's tested scientific stack.
    sys.path.append(str(CELLPOSE_OVERLAY))
    from cellpose import models  # noqa: PLC0415

    return models, library


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-id", required=True)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--model", choices=("cpsam_v2", "cpdino", "cpdino-vitb"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    if "000004" in arguments.crop_id or "000004" in str(arguments.raw):
        raise ValueError("MV-GTVOL-000004 is protected regression-only and cannot enter this bake-off")
    if arguments.output.exists():
        raise FileExistsError("Refusing to overwrite an immutable Cellpose run directory")
    if not torch.cuda.is_available():
        raise RuntimeError("This bake-off requires the already-qualified AMD ROCm Torch device")

    raw = np.load(arguments.raw, allow_pickle=False)
    if raw.ndim != 3 or not np.isfinite(raw).all():
        raise ValueError("Expected a finite, raw ZYX DVID array")

    models, compatibility_library = import_cellpose(arguments.model)
    checkpoint = CELLPOSE_MODELS / arguments.model
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Pinned checkpoint has not been acquired: {checkpoint}")

    arguments.output.mkdir(parents=True)
    model = models.CellposeModel(
        pretrained_model=arguments.model,
        device=torch.device("cuda"),
        # Float32 is deliberate: the ROCm compatibility gate was qualified in
        # float32, whereas bfloat16 support is hardware/runtime dependent.
        use_bfloat16=False,
    )
    model.net.eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    # Use upstream defaults, not a target-tuned tile parameter: SAM position
    # embeddings require 256; CellposeDINO's documented default is 384.
    tile_size = 256 if arguments.model.startswith("cpsam") else 384
    # do_3D=True is Cellpose's documented volumetric mode. compute_masks=False
    # freezes evaluation before Cellpose-specific dynamics or thresholds.
    _masks, flows, _styles = model.eval(
        raw,
        z_axis=0,
        do_3D=True,
        compute_masks=False,
        normalize=True,
        batch_size=1,
        bsize=tile_size,
    )
    torch.cuda.synchronize()
    flows_zyx = np.asarray(flows[1], dtype=np.float32)
    cell_probability_zyx = np.asarray(flows[2], dtype=np.float32)
    if not np.isfinite(flows_zyx).all() or not np.isfinite(cell_probability_zyx).all():
        raise RuntimeError("Cellpose returned non-finite raw network output")

    flow_path = arguments.output / "cellpose_flows_czyx.npy"
    probability_path = arguments.output / "cellpose_cell_probability_zyx.npy"
    np.save(flow_path, flows_zyx, allow_pickle=False)
    np.save(probability_path, cell_probability_zyx, allow_pickle=False)
    receipt = {
        "id": f"MV-CELLPOSE-ZEROSHOT-{arguments.crop_id}-{arguments.model}",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "COMPLETED_MACHINE_DIAGNOSTIC_ONLY",
        "classification": "INDEPENDENT_GENERAL_CELL_SEGMENTATION_MODEL_ZERO_SHOT_OUTPUT",
        "crop_id": arguments.crop_id,
        "protected_regression_used": False,
        "input": {"raw": str(arguments.raw.resolve()), "sha256": sha256(arguments.raw), "statistics": statistics(raw)},
        "model": {
            "name": arguments.model,
            "source": str(CELLPOSE_SOURCE.resolve()),
            "source_commit": os.popen(f'git -c safe.directory="{CELLPOSE_SOURCE}" -C "{CELLPOSE_SOURCE}" rev-parse HEAD').read().strip(),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_semantics": "Cellpose flow and cell-probability head; not Z/Y/X neuronal affinities",
        },
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "compatibility_adapter": "DECLARED_TORCHVISION_NMS_SCHEMA_ONLY; no NMS implementation or invocation",
            "normalization": "Cellpose eval(normalize=True): documented percentile normalization",
            "do_3D": True,
            "compute_masks": False,
            "tile_size": tile_size,
            "runtime_seconds": time.time() - started,
            "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "outputs": {
            "flows_czyx": {"path": str(flow_path.resolve()), "sha256": sha256(flow_path), "statistics": statistics(flows_zyx)},
            "cell_probability_zyx": {"path": str(probability_path.resolve()), "sha256": sha256(probability_path), "statistics": statistics(cell_probability_zyx)},
        },
        "scientific_boundary": "No Cellpose mask dynamics were run. Outputs have not been thresholded, interpreted as neurons, converted to labels, or used for training.",
    }
    (arguments.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    # Keep the library alive through the import and inference lifetime.
    _ = compatibility_library
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
