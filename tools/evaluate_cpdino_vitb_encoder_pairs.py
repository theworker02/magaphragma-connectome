"""Audit pinned CellposeDINO ViT-B encoder features at frozen reviewed pairs.

This intentionally evaluates the encoder, not Cellpose's flow/cell-probability
heads.  It is an exploratory, 2-D-encoder representation test: each Z plane is
processed with the documented Cellpose percentile normalization and the exact
CPDINO patch-embedding/transformer path.  It neither builds masks nor creates
neuronal affinities, labels, instances, or biological records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[1]
CELLPOSE_SOURCE = ROOT / "third_party" / "cellpose"
DINOV3_SOURCE = ROOT / "third_party" / "dinov3"
CELLPOSE_OVERLAY = ROOT / "third_party" / "cellpose-runtime" / "site-packages"
CELLPOSE_MODELS = ROOT / "third_party" / "cellpose-runtime" / "models"


def sha256(path: Path) -> str:
    """Return a content hash for all frozen inputs and the resulting receipt."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auc_different_high(same: list[float], different: list[float]) -> float:
    """Compute tie-aware pairwise AUC without adding a statistics dependency."""
    wins = 0.0
    for different_value in different:
        for same_value in same:
            wins += 1.0 if different_value > same_value else 0.5 if different_value == same_value else 0.0
    return wins / (len(same) * len(different))


def feature_summary(same: list[float], different: list[float]) -> dict[str, object]:
    """Keep both orientations so no post-hoc direction can be silently chosen."""
    same_array = np.asarray(same, dtype=np.float64)
    different_array = np.asarray(different, dtype=np.float64)
    high = auc_different_high(same, different)
    return {
        "same": {"count": len(same), "mean": float(same_array.mean()), "median": float(np.median(same_array))},
        "different": {"count": len(different), "mean": float(different_array.mean()), "median": float(np.median(different_array))},
        "auc_without_posthoc_orientation": {"different_high_auc": high, "different_low_auc": 1.0 - high},
    }


def load_cellpose() -> tuple[object, object, object]:
    """Load the pinned source plus overlay without mutating the SegNeuron venv.

    The Windows torchvision wheel lacks the NMS operator for this custom ROCm
    Torch build.  Cellpose needs transforms here, not NMS; declaring the schema
    only permits importing torchvision and never supplies or invokes NMS.
    """
    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(CELLPOSE_MODELS)
    schema_library = torch.library.Library("torchvision", "DEF")
    schema_library.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    sys.path.insert(0, str(DINOV3_SOURCE))
    sys.path.insert(0, str(CELLPOSE_SOURCE))
    sys.path.append(str(CELLPOSE_OVERLAY))
    from cellpose import models, transforms  # noqa: PLC0415

    return models, transforms, schema_library


def encoder_tokens(network: torch.nn.Module, image_chw: np.ndarray) -> torch.Tensor:
    """Execute CPDINO's documented encoder path through its final token norm.

    This duplicates only the pre-readout portion of CPDINO.forward so the
    retained result is the ViT feature token rather than a Cellpose head output.
    The checkpoint architecture and every operation match the pinned upstream
    CPDINO.forward implementation; the source code itself is not modified.
    """
    tensor = torch.from_numpy(image_chw[None]).to(network.device, dtype=network.dtype)
    with torch.no_grad():
        tensor = functional.conv2d(
            tensor,
            network.encoder.patch_embed.proj.weight.data[:, : tensor.shape[1]],
            bias=network.encoder.patch_embed.proj.bias.data,
            stride=network.encoder.patch_embed.proj.stride,
            padding=network.encoder.patch_embed.proj.padding,
        )
        height, width = tensor.shape[-2:]
        tensor = network.encoder.patch_embed.norm(tensor.flatten(2).transpose(1, 2))
        batch = tensor.shape[0]
        tensor = torch.cat(
            [network.encoder.cls_token.expand(batch, -1, -1), network.encoder.storage_tokens.expand(batch, -1, -1), tensor],
            axis=1,
        )
        rope = network.encoder.rope_embed(H=height, W=width)
        for block in network.encoder.blocks:
            tensor = block(tensor, rope)
        # CPDINO.forward removes one CLS plus four storage tokens before the
        # linear Cellpose readout.  We retain precisely those remaining tokens.
        return network.encoder.norm(tensor)[:, 5:].reshape(batch, height, width, -1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-receipt", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if "000004" in str(arguments.region_receipt) or "000004" in str(arguments.raw):
        raise ValueError("MV-GTVOL-000004 is protected and cannot enter the representation audit")
    if arguments.output.exists():
        raise FileExistsError("Refusing to overwrite an immutable encoder-feature receipt")
    if not torch.cuda.is_available():
        raise RuntimeError("The qualified AMD ROCm device is required")

    receipt = json.loads(arguments.region_receipt.read_text(encoding="utf-8"))
    raw = np.load(arguments.raw, allow_pickle=False)
    if raw.ndim != 3 or not np.isfinite(raw).all():
        raise ValueError("Expected a finite raw ZYX DVID array")
    raw_hash = sha256(arguments.raw)
    if raw_hash != receipt["raw_sha256"]:
        raise ValueError("Raw content hash does not match the frozen reviewed-region receipt")

    pairs = receipt["eligible_pairs"]
    if not pairs or any(pair["decision"] not in {"SAME_PROCESS", "DIFFERENT_PROCESS"} for pair in pairs):
        raise ValueError("Frozen receipt must contain only eligible SAME/DIFFERENT pairs")
    # Each requested coordinate must be interior to the unpadded raw volume.
    for pair in pairs:
        for point in (pair["pair_left_zyx"], pair["pair_right_zyx"]):
            if any(value < 0 or value >= limit for value, limit in zip(point, raw.shape, strict=True)):
                raise ValueError(f"Out-of-bounds reviewed coordinate: {point}")

    models, transforms, schema_library = load_cellpose()
    model = models.CellposeModel(pretrained_model="cpdino-vitb", device=torch.device("cuda"), use_bfloat16=False)
    network = model.net.eval()
    if network.model_name != "vitb" or network.ps != 8:
        raise ValueError("Unexpected CPDINO ViT-B architecture for this frozen audit")

    # Match Cellpose's documented 3-D pre-normalization, but intentionally run
    # each XY plane through the 2-D encoder.  This limitation is explicit in
    # the output and prevents treating the audit as a volumetric feature model.
    normalized = transforms.normalize_img(raw[..., None], normalize=True, norm3D=True)
    _, height, width = normalized.shape[:3]
    ypad1, ypad2, xpad1, xpad2 = transforms.get_pad_yx(height, width, min_size=(384, 384))
    padded_height, padded_width = height + ypad1 + ypad2, width + xpad1 + xpad2
    if (padded_height, padded_width) != (384, 384):
        raise ValueError("This fixed audit expects one documented 384x384 ViT-B tile per XY slice")

    required_z = sorted({point[0] for pair in pairs for point in (pair["pair_left_zyx"], pair["pair_right_zyx"])})
    # Feature maps are held only for Z slices containing reviewed endpoints;
    # this keeps the audit bounded and does not generate a dense DVID feature volume.
    token_maps: dict[int, np.ndarray] = {}
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for z_index in required_z:
        plane = normalized[z_index].transpose(2, 0, 1)
        plane = np.pad(plane, ((0, 0), (ypad1, ypad2), (xpad1, xpad2)), mode="constant")
        token_maps[z_index] = encoder_tokens(network, plane).squeeze(0).float().cpu().numpy()
    torch.cuda.synchronize()

    stride_y, stride_x = network.encoder.patch_embed.proj.stride
    pad_y, pad_x = network.encoder.patch_embed.proj.padding
    kernel_y, kernel_x = network.encoder.patch_embed.proj.kernel_size

    def point_feature(point: list[int]) -> np.ndarray:
        """Bilinearly interpolate CPDINO tokens at the raw pixel location.

        Nearest-token sampling makes every adjacent XY pair identical for an
        8-pixel token stride.  Interpolation preserves the encoder's native
        spatial resolution while avoiding a fabricated pixel-level feature map.
        """
        z_index, y_index, x_index = point
        padded_y, padded_x = y_index + ypad1, x_index + xpad1
        # Conv token center = token*stride - conv_padding + (kernel-1)/2.
        token_y = (padded_y + pad_y - (kernel_y - 1) / 2) / stride_y
        token_x = (padded_x + pad_x - (kernel_x - 1) / 2) / stride_x
        feature_map = token_maps[z_index]
        if not (0 <= token_y <= feature_map.shape[0] - 1 and 0 <= token_x <= feature_map.shape[1] - 1):
            raise ValueError("Reviewed point maps outside the CPDINO token grid")
        y0, x0 = int(np.floor(token_y)), int(np.floor(token_x))
        y1, x1 = min(y0 + 1, feature_map.shape[0] - 1), min(x0 + 1, feature_map.shape[1] - 1)
        wy, wx = token_y - y0, token_x - x0
        return ((1 - wy) * (1 - wx) * feature_map[y0, x0] + (1 - wy) * wx * feature_map[y0, x1]
                + wy * (1 - wx) * feature_map[y1, x0] + wy * wx * feature_map[y1, x1])

    rows: list[dict[str, object]] = []
    features: dict[str, dict[str, list[float]]] = {"encoder_l2_delta": {"SAME_PROCESS": [], "DIFFERENT_PROCESS": []}, "encoder_cosine_similarity": {"SAME_PROCESS": [], "DIFFERENT_PROCESS": []}}
    axis_decisions: dict[int, set[str]] = {0: set(), 1: set(), 2: set()}
    for pair in pairs:
        left = point_feature(pair["pair_left_zyx"])
        right = point_feature(pair["pair_right_zyx"])
        l2_delta = float(np.linalg.norm(left - right))
        cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
        decision = pair["decision"]
        axis_decisions[pair["channel_zyx"]].add(decision)
        features["encoder_l2_delta"][decision].append(l2_delta)
        features["encoder_cosine_similarity"][decision].append(cosine)
        rows.append({"interface_id": pair["interface_id"], "decision": decision, "channel_zyx": pair["channel_zyx"], "left_zyx": pair["pair_left_zyx"], "right_zyx": pair["pair_right_zyx"], "encoder_l2_delta": l2_delta, "encoder_cosine_similarity": cosine})

    checkpoint = CELLPOSE_MODELS / "cpdino-vitb"
    comparable_axes = sorted(axis for axis, decisions in axis_decisions.items() if decisions == {"SAME_PROCESS", "DIFFERENT_PROCESS"})
    output = {
        "id": f"MV-CELLPOSE-CPDINOVITB-ENCODER-PAIR-AUDIT-{receipt['crop_id']}",
        "status": "EXPLORATORY_INTERNAL_FEATURE_PAIR_SEPARABILITY_ONLY",
        "region": receipt["crop_id"],
        "region_receipt": {"path": str(arguments.region_receipt.resolve()), "sha256": sha256(arguments.region_receipt)},
        "raw": {"path": str(arguments.raw.resolve()), "sha256": raw_hash, "shape_zyx": list(raw.shape)},
        "model": {"name": "cpdino-vitb", "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint), "source_commit": os.popen(f'git -c safe.directory="{CELLPOSE_SOURCE}" -C "{CELLPOSE_SOURCE}" rev-parse HEAD').read().strip()},
        "feature_contract": {"network_stage": "CPDINO final encoder.norm tokens after CLS/storage-token removal and before Cellpose linear readout", "dim": int(next(iter(token_maps.values())).shape[-1]), "spatial_mode": "XY 2-D encoder features only; Z-adjacent pairs compare independently encoded neighboring XY planes", "normalization": "Cellpose transforms.normalize_img(normalize=True, norm3D=True)", "tile": {"shape_yx": [384, 384], "stride_yx": [stride_y, stride_x], "conv_padding_yx": [pad_y, pad_x], "raw_padding_yx": [ypad1, ypad2, xpad1, xpad2]}},
        "runtime": {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "torch_hip": torch.version.hip, "required_z_slices": required_z, "runtime_seconds": time.time() - started, "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated())},
        "features": {"encoder_l2_delta": feature_summary(features["encoder_l2_delta"]["SAME_PROCESS"], features["encoder_l2_delta"]["DIFFERENT_PROCESS"]), "encoder_cosine_similarity": feature_summary(features["encoder_cosine_similarity"]["SAME_PROCESS"], features["encoder_cosine_similarity"]["DIFFERENT_PROCESS"])},
        "axis_confounding_gate": {"class_presence_by_channel_zyx": {str(axis): sorted(decisions) for axis, decisions in axis_decisions.items()}, "axes_with_both_reviewed_classes": comparable_axes, "global_cross_class_auc_interpretable": bool(comparable_axes), "reason": "Global SAME/DIFFERENT AUC is not eligible for model selection when no affinity axis contains both reviewed classes; otherwise physical axis can explain class differences."},
        "pairs": rows,
        "scientific_boundary": "This is an exploratory 2-D general-cell encoder representation audit. It is not a documented neuronal-affinity output, does not make biological claims, and cannot enter training, labels, MV-FRAG, MV-N, MV-SYN, or MV-CONN without an independently predeclared and validated downstream gate.",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    # Retain the schema object until import/model lifetime completes.
    _ = schema_library
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
