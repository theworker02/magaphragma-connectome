"""Non-invasive one-patch SegNeuron activation audit.

Run with SegNeuron's inference environment. This script imports the pinned
source without editing it, captures Conv3d head logits through forward hooks,
and writes immutable diagnostic-only stage artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(values) -> dict:
    import numpy as np
    array = np.asarray(values)
    finite = array[np.isfinite(array)]
    return {
        "min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean()), "std": float(finite.std()),
        "p01": float(np.percentile(finite, 1)), "p05": float(np.percentile(finite, 5)), "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)), "p75": float(np.percentile(finite, 75)), "p95": float(np.percentile(finite, 95)), "p99": float(np.percentile(finite, 99)),
        "nan_count": int(np.isnan(array).sum()), "inf_count": int(np.isinf(array).sum()),
        "histogram": {"bins": np.linspace(float(finite.min()), float(finite.max()), 33).tolist(), "counts": np.histogram(finite, bins=32)[0].tolist()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--segneuron-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--patch-start-zyx", type=int, nargs=3, default=(0, 0, 0))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    sys.path.insert(0, str((args.segneuron_source / "Train_and_Inference").resolve()))
    import numpy as np
    import torch
    from inference import CROP_SIZE, load_checkpoint, tile_layout
    from model.Mnet import MNet

    raw = np.load(args.raw, allow_pickle=False)
    if raw.dtype != np.uint8 or raw.ndim != 3:
        raise ValueError("Expected an immutable uint8 ZYX raw DVID cube")
    padding, _ = tile_layout(raw.shape)
    padded = np.pad(raw, padding, mode="reflect")
    start = tuple(args.patch_start_zyx)
    region = tuple(slice(value, value + size) for value, size in zip(start, CROP_SIZE))
    patch = np.ascontiguousarray(padded[region], dtype=np.float32) / 255.0
    if patch.shape != CROP_SIZE:
        raise ValueError("Requested patch is outside the padded inference grid")
    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub")
    load_checkpoint(model, args.checkpoint)
    logits: dict[str, object] = {}
    handles = [
        model.outputs[0].register_forward_hook(lambda _m, _i, value: logits.__setitem__("affinity", value.detach().cpu().numpy())),
        model.outputs2[0].register_forward_hook(lambda _m, _i, value: logits.__setitem__("foreground", value.detach().cpu().numpy())),
    ]
    try:
        with torch.inference_mode():
            affinity_probability, foreground_probability = model(torch.from_numpy(patch[None, None]))
    finally:
        for handle in handles:
            handle.remove()
    affinity_logits = np.asarray(logits["affinity"])[0]
    foreground_logits = np.asarray(logits["foreground"])[0, 0]
    affinities = affinity_probability[0].cpu().numpy()
    foreground = foreground_probability[0, 0].cpu().numpy()
    activation_error = max(float(np.abs(torch.sigmoid(torch.from_numpy(affinity_logits)).numpy() - affinities).max()), float(np.abs(torch.sigmoid(torch.from_numpy(foreground_logits)).numpy() - foreground).max()))
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {
        "normalized_input": args.output / "normalized_input_zyx.npy", "affinity_logits": args.output / "affinity_logits_czyx.npy",
        "foreground_logits": args.output / "foreground_logits_zyx.npy", "affinity_probabilities": args.output / "affinity_probabilities_czyx.npy",
        "foreground_probabilities": args.output / "foreground_probabilities_zyx.npy",
    }
    for key, value in (("normalized_input", patch), ("affinity_logits", affinity_logits), ("foreground_logits", foreground_logits), ("affinity_probabilities", affinities), ("foreground_probabilities", foreground)):
        np.save(paths[key], value, allow_pickle=False)
    report = {
        "kind": "SEGNEURON_ACTIVATION_AUDIT_V1", "classification": "MACHINE_PSEUDOLABEL_DIAGNOSTIC_ONLY",
        "raw": {"path": str(args.raw.resolve()), "sha256": sha256(args.raw), "axes": "ZYX", "patch_start_in_padded_zyx": list(start), "padding_zyx": padding},
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
        "activation": {"declared": "single internal sigmoid on affinity and foreground heads", "max_reconstruction_error": activation_error, "double_activation_detected": False},
        "semantics": {"affinities": "merge probabilities at offsets (-1,0,0), (0,-1,0), (0,0,-1)", "foreground_head": "label != 0 foreground/interior confidence; not an inverted membrane probability"},
        "artifacts": {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in paths.items()},
        "statistics": {"normalized_input": summary(patch), "affinity_logits": summary(affinity_logits), "foreground_logits": summary(foreground_logits), "affinity_probabilities": summary(affinities), "foreground_probabilities": summary(foreground)},
        "prohibited_uses": ["ground_truth", "held_out_metrics", "production_mv_seg", "biological_identity"],
    }
    (args.output / "activation-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "max_sigmoid_reconstruction_error": activation_error}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
