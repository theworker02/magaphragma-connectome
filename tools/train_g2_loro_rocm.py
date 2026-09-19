"""Bounded, fold-isolated G2 LORO adaptation from the frozen G1 checkpoint.

Each invocation trains exactly one fold.  It reads a frozen G2 dataset manifest,
uses only the three permitted training regions, and reports all held-out and
independent validation pair metrics without using them in the optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

ROOT = pathlib.Path(__file__).resolve().parents[1]
_g1_spec = importlib.util.spec_from_file_location("g1_trainer", ROOT / "tools" / "train_dvid_adaptation_rocm.py")
assert _g1_spec and _g1_spec.loader
g1 = importlib.util.module_from_spec(_g1_spec); _g1_spec.loader.exec_module(g1)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ManifestPairs:
    """A reviewed pair source limited to the frozen interior-pair allow-list."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self.crop_id = str(record["crop_id"])
        raw_path = g1.local_path(record["raw"]["path"])
        if g1.sha256(raw_path) != record["raw"]["sha256"]:
            raise ValueError(f"{self.crop_id}: raw crop hash no longer matches frozen G2 dataset")
        self.raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        self.targets = np.load(g1.local_path(record["targets"]["path"]), mmap_mode="r", allow_pickle=False)
        self.mask = np.load(g1.local_path(record["mask"]["path"]), mmap_mode="r", allow_pickle=False)
        self.entries = [(int(e["channel_zyx"]), *[int(v) for v in e["pair_left_zyx"]]) for e in record["usable_interior_pairs"]]
        if not self.entries:
            raise ValueError(f"{self.crop_id}: no admitted interior reviewed pairs")
        self.same = sum(int(self.targets[item]) == 1 for item in self.entries)
        self.different = len(self.entries) - self.same
        if not self.same or not self.different:
            raise ValueError(f"{self.crop_id}: requires both SAME and DIFFERENT pairs")

    def patch_for(self, entry: tuple[int, int, int, int]):
        channel, z, y, x = entry
        origin = tuple(max(0, min(value - extent // 2, limit - extent)) for value, extent, limit in zip((z, y, x), g1.PATCH_ZYX, self.raw.shape, strict=True))
        slices = tuple(slice(start, start + extent) for start, extent in zip(origin, g1.PATCH_ZYX, strict=True))
        raw = np.asarray(self.raw[slices], dtype=np.float32) / 255.0
        targets = np.asarray(self.targets[(slice(None),) + slices], dtype=np.float32)
        full_mask = np.zeros_like(targets, dtype=np.float32)
        # Preserve only manifest-approved pairs in this patch; the materialized
        # tensor remains immutable evidence but crop-edge pairs are ignore.
        for c, ez, ey, ex in self.entries:
            if all(start <= point < start + extent for point, start, extent in zip((ez, ey, ex), origin, g1.PATCH_ZYX, strict=True)):
                full_mask[c, ez - origin[0], ey - origin[1], ex - origin[2]] = 1.0
        local = (channel, z - origin[0], y - origin[1], x - origin[2])
        if full_mask[local] != 1:
            raise RuntimeError("Frozen reviewed anchor disappeared from G2 patch")
        return raw, targets, full_mask, {"crop_id": self.crop_id, "anchor_channel_zyx": channel, "anchor_zyx": [z, y, x], "patch_origin_zyx": list(origin), "supervised_pairs_in_patch": int(full_mask.sum())}


def metrics(model: torch.nn.Module, pairs: ManifestPairs) -> dict[str, Any]:
    was_training = model.training; model.eval(); rows: list[tuple[int, float]] = []; losses: list[float] = []
    with torch.no_grad():
        for entry in pairs.entries:
            raw, targets, mask, metadata = pairs.patch_for(entry)
            affinity, _ = model(torch.from_numpy(raw[None, None]).cuda())
            truth = torch.from_numpy(targets[None]).cuda(); review_mask = torch.from_numpy(mask[None]).cuda()
            losses.append(float(g1.masked_affinity_bce(affinity, truth, review_mask).cpu()))
            channel, z, y, x = entry; oz, oy, ox = metadata["patch_origin_zyx"]
            rows.append((int(targets[channel, z - oz, y - oy, x - ox]), float(affinity[0, channel, z - oz, y - oy, x - ox].cpu())))
    model.train(was_training)
    same, different = [v for t, v in rows if t == 1], [v for t, v in rows if t == 0]
    q = lambda values: [float(v) for v in np.quantile(values, [0.1, 0.5, 0.9])]
    return {"reviewed_pairs": len(rows), "masked_affinity_bce": float(np.mean(losses)), "same_pair_count": len(same), "different_pair_count": len(different), "same_mean_affinity": float(np.mean(same)), "same_quantiles_p10_p50_p90": q(same), "different_mean_affinity": float(np.mean(different)), "different_quantiles_p10_p50_p90": q(different), "same_accuracy_at_0_5": float(np.mean(np.asarray(same) >= .5)), "different_accuracy_at_0_5": float(np.mean(np.asarray(different) < .5)), "margin": float(np.mean(same) - np.mean(different)), "pair_predictions": [{"target": t, "affinity": v} for t, v in rows]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--fold", choices=list("ABCD"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--target-loss-weight", type=float, default=.25)
    parser.add_argument("--resume-checkpoint", type=pathlib.Path, default=ROOT / "experiments/phase6c/MV-TRAIN-SEGNEURON-DVID-G1-001-ATTEMPT-002/checkpoint-step2280.pt")
    args = parser.parse_args()
    if args.steps <= 0 or args.checkpoint_every <= 0 or not 0 < args.target_loss_weight <= 1: raise ValueError("Invalid bounded G2 schedule")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if dataset.get("status") != "FROZEN_READY_FOR_G2_LORO" or dataset.get("protected_regression", {}).get("included"):
        raise ValueError("G2 dataset is not frozen or contaminates the protected regression")
    held = f"MV-G2-TRAIN-{args.fold}"
    records = {str(r["crop_id"]): r for r in dataset["regions"]}
    fold = next((f for f in dataset["loro_folds"] if f["held_out_region"] == held), None)
    if fold is None: raise ValueError("Missing frozen LORO fold")
    train_sets = [ManifestPairs(records[name]) for name in fold["train_with"]]
    held_pairs = ManifestPairs(records[held])
    independent = [ManifestPairs(records[f"MV-G2-VALIDATION-V{index}"]) for index in (1, 2, 3)]
    output = ROOT / "experiments/phase6d" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    model = g1.MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub").cuda().train(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    prior = torch.load(args.resume_checkpoint, map_location="cuda", weights_only=False)
    if not {"model_weights", "optimizer", "step"}.issubset(prior): raise ValueError("G1 checkpoint lacks exact model/Adam state")
    model.load_state_dict(prior["model_weights"], strict=True); optimizer.load_state_dict(prior["optimizer"]); start = int(prior["step"])
    source_root = ROOT / "datasets/cache/segneuron/EMNeuron-labeled/labeled/Hemi-brain-fib"; source_loss = g1.WeightedBCE(); frozen_source = g1.source_validation(source_root)
    history: list[dict[str, Any]] = []; checkpoints: list[dict[str, Any]] = []; started = time.time(); best: dict[str, Any] | None = None
    for index in range(args.steps):
        global_step = start + index + 1; source_raw, source_aff, source_fg, source_meta = g1.source_batch(source_root, start + index)
        target_set = train_sets[index % len(train_sets)]; target_entry = target_set.entries[(index // len(train_sets)) % len(target_set.entries)]
        raw, target, mask, target_meta = target_set.patch_for(target_entry)
        source_aff_pred, source_fg_pred = model(source_raw); target_pred, _ = model(torch.from_numpy(raw[None, None]).cuda())
        source_value = source_loss(source_aff_pred, source_aff) + source_loss(source_fg_pred, source_fg)
        target_value = g1.masked_affinity_bce(target_pred, torch.from_numpy(target[None]).cuda(), torch.from_numpy(mask[None]).cuda())
        total = source_value + args.target_loss_weight * target_value; optimizer.zero_grad(); total.backward(); optimizer.step(); torch.cuda.synchronize()
        if not np.isfinite(float(total.detach().cpu())): raise RuntimeError(f"Non-finite G2 loss at step {global_step}")
        history.append({"global_step":global_step,"source_loss":float(source_value.detach().cpu()),"target_masked_affinity_loss":float(target_value.detach().cpu()),"total_loss":float(total.detach().cpu()),"source_batch":source_meta,"target_batch":target_meta,"gpu_memory_allocated_bytes":int(torch.cuda.memory_allocated())})
        if global_step % args.checkpoint_every == 0 or index + 1 == args.steps:
            checkpoint = output / f"checkpoint-step{global_step}.pt"; torch.save({"model_weights":model.state_dict(),"optimizer":optimizer.state_dict(),"step":global_step,"parent_checkpoint":str(args.resume_checkpoint.resolve()),"training_classification":"EXPERIMENTAL_G2_LORO_REVIEWED_DVID_TARGET_ADAPTATION","fold":args.fold}, checkpoint)
            model.eval()
            with torch.no_grad():
                s_aff, s_fg = model(frozen_source[0]); source_val=float((source_loss(s_aff,frozen_source[1])+source_loss(s_fg,frozen_source[2])).cpu())
            model.train(); held_metrics=metrics(model,held_pairs); independent_metrics={pairs.crop_id:metrics(model,pairs) for pairs in independent}
            record={"global_step":global_step,"checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":sha256(checkpoint),"source_validation_loss":source_val,"held_out_region":held_metrics,"independent_validation":independent_metrics}
            checkpoints.append(record)
            # Selection uses held-out pair loss only; reconstruction is never an optimization signal.
            if best is None or float(held_metrics["masked_affinity_bce"]) < float(best["held_out_region"]["masked_affinity_bce"]): best=record
            (output / "progress.json").write_text(json.dumps({"status":"RUNNING","history":history,"checkpoints":checkpoints,"best_held_out_pair_checkpoint":best,"protected_regression_used":False},indent=2)+"\n")
    receipt={"status":"COMPLETED","classification":"EXPERIMENTAL_G2_LORO_REVIEWED_DVID_TARGET_ADAPTATION","run_id":args.run_id,"fold":args.fold,"held_out_region":held,"train_regions":fold["train_with"],"start_global_step":start,"final_global_step":start+args.steps,"resume":{"checkpoint":str(args.resume_checkpoint.resolve()),"sha256":sha256(args.resume_checkpoint),"exact_model_and_adam_state_restored":True,"scheduler":"NONE"},"dataset":{"path":str(args.dataset.resolve()),"sha256":sha256(args.dataset)},"target_loss_weight":args.target_loss_weight,"source_supervision":{"cohort":"Hemi-brain-fib/0..3","frozen_validation":"Hemi-brain-fib/4"},"history":history,"checkpoints":checkpoints,"best_held_out_pair_checkpoint":best,"device":torch.cuda.get_device_name(0),"torch_version":torch.__version__,"torch_hip_version":torch.version.hip,"peak_gpu_memory_allocated_bytes":int(torch.cuda.max_memory_allocated()),"runtime_seconds":time.time()-started,"protected_regression_used":False,"biological_promotions":{"MV-FRAG":0,"MV-N":0,"MV-SYN":0,"MV-CONN":0}}
    (output / "receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__ == "__main__": main()
