"""Run the audited PyTorch Connectomics checkpoint on a frozen DVID dev crop.

This is a technical independent-evidence smoke test.  It writes raw network
logits and probabilities only; it never thresholds, creates labels, or makes
biological records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from connectomics.config.schema.root import Config
from connectomics.models import build_model


PROTECTED_PARENT = "MV-GTVOL-000004"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_config() -> object:
    return OmegaConf.create(
        {
            "model": {
                "arch": {"type": "mednext"},
                "in_channels": 1,
                "out_channels": 6,
                "loss": {"deep_supervision": False},
                "mednext": {
                    "size": "L",
                    "kernel_size": 3,
                    "dim": "3d",
                    "checkpoint_style": "outside_block",
                },
            }
        }
    )


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    # Checkpoint is from the official PyTC Hugging Face repository.  Keep
    # weights-only deserialization and allowlist only PyTC's configuration
    # dataclass; never fall back to unsafe pickle execution.
    torch.serialization.add_safe_globals([Config])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(model_config())
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint architecture mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return model.eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-id", required=True)
    parser.add_argument("--raw-zyx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--origin-zyx", nargs=3, type=int, required=True)
    parser.add_argument("--shape-zyx", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.parent_id == PROTECTED_PARENT:
        raise ValueError("MV-GTVOL-000004 is regression-only and cannot supply this smoke input")
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {arguments.output}")

    parent = np.load(arguments.raw_zyx, mmap_mode="r", allow_pickle=False)
    origin = tuple(arguments.origin_zyx)
    shape = tuple(arguments.shape_zyx)
    if parent.ndim != 3 or any(start < 0 or start + length > limit for start, length, limit in zip(origin, shape, parent.shape, strict=True)):
        raise ValueError("requested crop is outside the immutable ZYX parent array")
    crop_zyx = np.asarray(parent[tuple(slice(start, start + length) for start, length in zip(origin, shape, strict=True))]).copy()
    if crop_zyx.shape != shape or crop_zyx.dtype != np.uint8:
        raise ValueError("unexpected DVID crop shape or dtype")

    arguments.output.mkdir(parents=True)
    raw_path = arguments.output / "raw_source_zyx.npy"
    np.save(raw_path, crop_zyx, allow_pickle=False)

    # PyTC BANIS documents XYZ source ordering.  Preserve the source ZYX crop
    # separately and make this conversion explicit and reversible.
    input_xyz = np.transpose(crop_zyx, (2, 1, 0)).copy()
    device = torch.device("cuda")
    model = load_model(arguments.checkpoint).to(device)
    tensor = torch.from_numpy(input_xyz.astype(np.float32, copy=False) / 255.0)[None, None].to(device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        logits_cxyz = model(tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # The network emits BCXYZ.  A single smoke input has B=1; only after
    # explicitly removing that batch axis can channel/spatial axes be mapped
    # back to the project-standard CZYX representation.
    logits_bcxyz_np = logits_cxyz.float().cpu().numpy()
    if logits_bcxyz_np.ndim != 5:
        raise RuntimeError(f"expected PyTC BCXYZ output, got shape {logits_bcxyz_np.shape}")
    if logits_bcxyz_np.shape[0] != 1:
        raise RuntimeError(
            "single-crop smoke inference requires exactly one batch item; "
            f"got B={logits_bcxyz_np.shape[0]}"
        )
    if logits_bcxyz_np.shape[1] != 6:
        raise RuntimeError(
            "audited BANIS checkpoint requires six affinity channels; "
            f"got C={logits_bcxyz_np.shape[1]}"
        )
    probabilities_czyx = np.transpose(
        1.0 / (1.0 + np.exp(-logits_bcxyz_np[0])), (0, 3, 2, 1)
    ).copy()
    if not np.isfinite(logits_bcxyz_np).all() or not np.isfinite(probabilities_czyx).all():
        raise RuntimeError("non-finite independent-model output")

    logits_path = arguments.output / "raw_logits_cxyz.npy"
    probabilities_path = arguments.output / "probabilities_czyx.npy"
    np.save(logits_path, logits_bcxyz_np, allow_pickle=False)
    np.save(probabilities_path, probabilities_czyx, allow_pickle=False)
    receipt = {
        "id": "MV-INDEP-PYTC-SMOKE-000001",
        "status": "MACHINE_PREDICTION_NOT_LABEL_OR_BIOLOGICAL_EVIDENCE",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model_id": "MV-INDEP-MODEL-PYTC-BANIS-001",
        "parent_id": arguments.parent_id,
        "source_origin_zyx": list(origin),
        "source_shape_zyx": list(shape),
        "source_raw_path": str(arguments.raw_zyx.resolve()),
        "source_raw_sha256": sha256(arguments.raw_zyx),
        "crop_raw_sha256": sha256(raw_path),
        "checkpoint_path": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": sha256(arguments.checkpoint),
        "normalization": "uint8 / 255.0",
        "model_input_layout": "CXYZ",
        "source_layout": "ZYX",
        "layout_transform": "source_zyx.transpose(2,1,0) -> xyz; assert B=1; output_bcxyz[0].transpose(0,3,2,1) -> czyx",
        "output_semantics": "six raw affinity logits; derived sigmoid probabilities, no thresholding",
        "raw_logits_path": str(logits_path.resolve()),
        "raw_logits_sha256": sha256(logits_path),
        "raw_logits_shape_bcxyz": list(logits_bcxyz_np.shape),
        "raw_logits_statistics": {"min": float(logits_bcxyz_np.min()), "max": float(logits_bcxyz_np.max()), "mean": float(logits_bcxyz_np.mean()), "std": float(logits_bcxyz_np.std()), "nan_count": int(np.isnan(logits_bcxyz_np).sum()), "inf_count": int(np.isinf(logits_bcxyz_np).sum())},
        "probabilities_path": str(probabilities_path.resolve()),
        "probabilities_sha256": sha256(probabilities_path),
        "probabilities_shape_czyx": list(probabilities_czyx.shape),
        "probabilities_statistics": {"min": float(probabilities_czyx.min()), "max": float(probabilities_czyx.max()), "mean": float(probabilities_czyx.mean()), "std": float(probabilities_czyx.std())},
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "runtime_seconds": elapsed,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "protected_region_excluded": True,
    }
    (arguments.output / "inference_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
