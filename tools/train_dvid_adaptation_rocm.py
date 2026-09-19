"""Run bounded G1 DVID adaptation from the preserved source step-2250 state.

The target term consumes only human-reviewed, masked Z/Y/X local affinity
pairs.  It deliberately does not treat the target crop as a dense instance
label volume, and it never reads MV-GTVOL-000004.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from functools import lru_cache
import re

import imageio
import numpy as np
import torch
import torch.nn.functional as functional

REPOSITORY = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(REPOSITORY / "third_party/segneuron/Train_and_Inference"))
from loss.loss import WeightedBCE
from model.Mnet import MNet
from utils.aff_util import seg_to_affgraph
from utils.seg_util import mknhood3d

PATCH_ZYX = (20, 128, 128)


def local_path(value: str) -> pathlib.Path:
    """Resolve immutable Windows provenance paths when this trainer runs in WSL.

    Receipts retain their original absolute Windows paths; this adapter affects
    only the local read operation and never rewrites provenance.
    """
    path = pathlib.Path(value)
    if path.exists():
        return path
    match = re.fullmatch(r"([A-Za-z]):\\(.*)", value)
    if match:
        candidate = pathlib.Path("/mnt") / match.group(1).lower() / match.group(2).replace("\\", "/")
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Receipt-referenced local source does not exist in this runtime: {value}")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def source_pair(root_text: str, source_id: int) -> tuple[np.ndarray, np.ndarray]:
    root = pathlib.Path(root_text)
    return imageio.volread(root / f"{source_id}.tif"), imageio.volread(root / f"{source_id}_MaskIns.tif")


def source_batch(root: pathlib.Path, iteration: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    source_id = iteration % 4
    raw, label = source_pair(str(root), source_id)
    z, y, x = (iteration * 13) % 400, (iteration * 29) % 390, (iteration * 37) % 390
    raw_patch = raw[z:z + 20, y:y + 128, x:x + 128].astype("float32") / 255.0
    label_patch = label[z:z + 20, y:y + 128, x:x + 128]
    affinity = seg_to_affgraph(label_patch, mknhood3d(1), pad="replicate").astype("float32")
    return (torch.from_numpy(raw_patch[None, None]).cuda(), torch.from_numpy(affinity[None]).cuda(),
            torch.from_numpy((label_patch != 0).astype("float32")[None, None]).cuda(),
            {"source_pair": f"Hemi-brain-fib/{source_id}", "patch_origin_zyx": [z, y, x]})


def source_validation(root: pathlib.Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = imageio.volread(root / "4.tif")[:20, :128, :128].astype("float32") / 255.0
    label = imageio.volread(root / "4_MaskIns.tif")[:20, :128, :128]
    affinity = seg_to_affgraph(label, mknhood3d(1), pad="replicate").astype("float32")
    return (torch.from_numpy(raw[None, None]).cuda(), torch.from_numpy(affinity[None]).cuda(),
            torch.from_numpy((label != 0).astype("float32")[None, None]).cuda())


class ReviewedPairs:
    def __init__(self, receipt_path: pathlib.Path) -> None:
        self.receipt_path = receipt_path.resolve()
        self.receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if self.receipt.get("status") != "REVIEWED_DVID_AFFINITY_SUPERVISION":
            raise ValueError("Target receipt is not reviewed affinity supervision")
        self.raw = np.load(local_path(self.receipt["raw"]["path"]), mmap_mode="r", allow_pickle=False)
        self.targets = np.load(local_path(self.receipt["targets"]["path"]), mmap_mode="r", allow_pickle=False)
        self.mask = np.load(local_path(self.receipt["mask"]["path"]), mmap_mode="r", allow_pickle=False)
        if self.raw.shape != tuple(self.receipt["raw"]["shape_zyx"]) or self.targets.shape != (3, *self.raw.shape) or self.mask.shape != self.targets.shape:
            raise ValueError("Reviewed target tensors have invalid geometry")
        self.entries = [(int(channel), *[int(value) for value in point]) for channel, *point in np.argwhere(self.mask)]
        if not self.entries:
            raise ValueError("Reviewed target receipt contains no supervised pairs")
        self.same = sum(int(self.targets[entry]) == 1 for entry in self.entries)
        self.different = len(self.entries) - self.same
        if not self.same or not self.different:
            raise ValueError("Reviewed target receipt needs both SAME and DIFFERENT pairs")

    def patch_for(self, entry: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        channel, z, y, x = entry
        point = (z, y, x)
        origin = tuple(max(0, min(value - extent // 2, limit - extent)) for value, extent, limit in zip(point, PATCH_ZYX, self.raw.shape, strict=True))
        slices = tuple(slice(start, start + extent) for start, extent in zip(origin, PATCH_ZYX, strict=True))
        raw = np.asarray(self.raw[slices], dtype=np.float32) / 255.0
        targets = np.asarray(self.targets[(slice(None),) + slices], dtype=np.float32)
        mask = np.asarray(self.mask[(slice(None),) + slices], dtype=np.float32)
        local = (channel, z - origin[0], y - origin[1], x - origin[2])
        if mask[local] != 1:
            raise RuntimeError("Target patch lost its anchor reviewed affinity pair")
        return raw, targets, mask, {"anchor_channel_zyx": channel, "anchor_zyx": [z, y, x], "patch_origin_zyx": list(origin),
                                    "anchor_decision": "SAME_PROCESS" if targets[local] == 1 else "DIFFERENT_PROCESS",
                                    "supervised_pairs_in_patch": int(mask.sum())}


def masked_affinity_bce(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = mask.bool()
    if not bool(selected.any()):
        raise ValueError("Masked target loss has no supervised affinity pairs")
    return functional.binary_cross_entropy(prediction, target, reduction="none")[selected].mean()


def target_metrics(model: torch.nn.Module, pairs: ReviewedPairs) -> dict[str, object]:
    was_training = model.training; model.eval()
    predictions: list[tuple[int, float]] = []
    losses: list[float] = []
    with torch.no_grad():
        for entry in pairs.entries:
            raw, targets, mask, metadata = pairs.patch_for(entry)
            affinity, _ = model(torch.from_numpy(raw[None, None]).cuda())
            target = torch.from_numpy(targets[None]).cuda(); target_mask = torch.from_numpy(mask[None]).cuda()
            losses.append(float(masked_affinity_bce(affinity, target, target_mask).cpu()))
            channel, z, y, x = entry
            predictions.append((int(targets[channel, z - metadata["patch_origin_zyx"][0], y - metadata["patch_origin_zyx"][1], x - metadata["patch_origin_zyx"][2]]),
                                float(affinity[0, channel, z - metadata["patch_origin_zyx"][0], y - metadata["patch_origin_zyx"][1], x - metadata["patch_origin_zyx"][2]].cpu())))
    model.train(was_training)
    same = [value for truth, value in predictions if truth == 1]
    different = [value for truth, value in predictions if truth == 0]
    return {"reviewed_pairs": len(predictions), "masked_affinity_bce": float(np.mean(losses)),
            "same_pair_count": len(same), "different_pair_count": len(different),
            "same_mean_affinity": float(np.mean(same)), "different_mean_affinity": float(np.mean(different)),
            "same_accuracy_at_0_5": float(np.mean(np.asarray(same) >= 0.5)),
            "different_accuracy_at_0_5": float(np.mean(np.asarray(different) < 0.5)),
            "pair_predictions": [{"target": truth, "affinity": value} for truth, value in predictions]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="MV-TRAIN-SEGNEURON-DVID-G1-001")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--target-loss-weight", type=float, default=0.25)
    parser.add_argument("--resume-checkpoint", type=pathlib.Path, default=REPOSITORY / "experiments/phase6b/MV-TRAIN-SEGNEURON-FIBSEM-003/checkpoint-step2250.pt")
    parser.add_argument("--train-receipt", type=pathlib.Path, default=REPOSITORY / "local_research_build/phase6e/external-review/dense-raw-candidate-001/reviewed-affinity-supervision-v2/receipt.json")
    parser.add_argument("--validation-receipt", type=pathlib.Path, default=REPOSITORY / "local_research_build/phase6e/external-review/dense-raw-candidate-002/reviewed-affinity-supervision-v2/receipt.json")
    args = parser.parse_args()
    if args.steps <= 0 or args.checkpoint_every <= 0 or not (0 < args.target_loss_weight <= 1):
        raise ValueError("steps/checkpoint interval must be positive and target loss weight must be in (0,1]")
    if "000004" in str(args.train_receipt) or "000004" in str(args.validation_receipt):
        raise ValueError("MV-GTVOL-000004 is regression-only and prohibited from G1 supervision")
    output = REPOSITORY / "experiments/phase6c" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    source_root = REPOSITORY / "datasets/cache/segneuron/EMNeuron-labeled/labeled/Hemi-brain-fib"
    train_pairs, validation_pairs = ReviewedPairs(args.train_receipt), ReviewedPairs(args.validation_receipt)
    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub").cuda().train(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    prior = torch.load(args.resume_checkpoint, map_location="cuda", weights_only=False)
    if not {"model_weights", "optimizer", "step"}.issubset(prior): raise ValueError("Resume checkpoint lacks model/Adam/global-step state")
    model.load_state_dict(prior["model_weights"], strict=True); optimizer.load_state_dict(prior["optimizer"]); start = int(prior["step"])
    source_loss = WeightedBCE(); frozen_source_validation = source_validation(source_root)
    started = time.time(); history: list[dict[str, object]] = []; checkpoints: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for index in range(args.steps):
        global_step = start + index + 1
        source_raw, source_affinity, source_foreground, source_metadata = source_batch(source_root, start + index)
        target_entry = train_pairs.entries[index % len(train_pairs.entries)]
        target_raw, target_affinity, target_mask, target_metadata = train_pairs.patch_for(target_entry)
        source_affinity_pred, source_foreground_pred = model(source_raw)
        target_affinity_pred, _ = model(torch.from_numpy(target_raw[None, None]).cuda())
        source_value = source_loss(source_affinity_pred, source_affinity) + source_loss(source_foreground_pred, source_foreground)
        target_value = masked_affinity_bce(target_affinity_pred, torch.from_numpy(target_affinity[None]).cuda(), torch.from_numpy(target_mask[None]).cuda())
        total = source_value + args.target_loss_weight * target_value
        optimizer.zero_grad(); total.backward(); optimizer.step(); torch.cuda.synchronize()
        if not np.isfinite(float(total.detach().cpu())): raise RuntimeError(f"Non-finite G1 loss at global step {global_step}")
        item = {"global_step": global_step, "source_loss": float(source_value.detach().cpu()), "target_masked_affinity_loss": float(target_value.detach().cpu()),
                "total_loss": float(total.detach().cpu()), "source_batch": source_metadata, "target_batch": target_metadata,
                "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated())}
        history.append(item)
        if global_step % args.checkpoint_every == 0 or index + 1 == args.steps:
            checkpoint = output / f"checkpoint-step{global_step}.pt"
            torch.save({"model_weights": model.state_dict(), "optimizer": optimizer.state_dict(), "step": global_step,
                        "parent_checkpoint": str(args.resume_checkpoint.resolve()), "training_classification": "EXPERIMENTAL_REVIEWED_DVID_TARGET_ADAPTATION"}, checkpoint)
            model.eval()
            with torch.no_grad():
                source_val_affinity, source_val_foreground = model(frozen_source_validation[0])
                source_val = float((source_loss(source_val_affinity, frozen_source_validation[1]) + source_loss(source_val_foreground, frozen_source_validation[2])).cpu())
            model.train(); target_val = target_metrics(model, validation_pairs)
            record = {"global_step": global_step, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
                      "source_validation_loss": source_val, "reviewed_dvid_validation": target_val}
            checkpoints.append(record)
            if best is None or float(target_val["masked_affinity_bce"]) < float(best["reviewed_dvid_validation"]["masked_affinity_bce"]): best = record
            (output / "progress.json").write_text(json.dumps({"status": "RUNNING", "history": history, "checkpoints": checkpoints,
                "best_reviewed_dvid_validation_checkpoint": best, "g1_policy": "Selection uses frozen reviewed DVID validation; MV-GTVOL-000004 not read."}, indent=2) + "\n", encoding="utf-8")
    receipt = {"status": "COMPLETED", "classification": "EXPERIMENTAL_REVIEWED_DVID_TARGET_ADAPTATION", "run_id": args.run_id,
               "start_global_step": start, "final_global_step": start + args.steps, "resume": {"checkpoint": str(args.resume_checkpoint.resolve()), "sha256": sha256(args.resume_checkpoint), "exact_model_and_adam_state_restored": True, "scheduler": "NONE"},
               "target_supervision": {"train_receipt": str(args.train_receipt.resolve()), "train_sha256": sha256(args.train_receipt), "validation_receipt": str(args.validation_receipt.resolve()), "validation_sha256": sha256(args.validation_receipt), "loss": "masked_binary_cross_entropy_on_reviewed_ZYX_pairs_only", "weight": args.target_loss_weight},
               "source_supervision": {"cohort": "Hemi-brain-fib/0..3", "frozen_validation": "Hemi-brain-fib/4"}, "history": history, "checkpoints": checkpoints,
               "best_reviewed_dvid_validation_checkpoint": best, "device": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "torch_hip_version": torch.version.hip,
               "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()), "runtime_seconds": time.time() - started,
               "regression_volume": "MV-GTVOL-000004", "regression_volume_used": False, "biological_promotions": {"MV-FRAG": 0, "MV-N": 0, "MV-SYN": 0, "MV-CONN": 0}}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
