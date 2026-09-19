"""Bounded, interface-normalized G3 LORO adaptation on the preserved ROCm stack.

Each fold restores the exact G1 step-2280 model and Adam state.  Target
interfaces, rather than individual correlated pairs, are sampled uniformly.
The held-out region and V1/V2/V3 are evaluation-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Reuse the already-qualified upstream SegNeuron/ROCm implementation.  This
# runner changes supervision selection only; it does not change MNet or loss
# semantics for the legitimate source FIB-SEM branch.
spec = importlib.util.spec_from_file_location("g1_trainer", ROOT / "tools" / "train_dvid_adaptation_rocm.py")
assert spec and spec.loader
g1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g1)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_bce(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # `weights` is zero outside reviewed pairs and sums to one per reviewed
    # interface.  The denominator keeps sparse patches on a comparable scale.
    selected = mask.bool()
    denominator = weights[selected].sum()
    if not bool(selected.any()) or float(denominator.detach().cpu()) <= 0:
        raise ValueError("G3 target patch has no positive supervised interface weight")
    return (functional.binary_cross_entropy(prediction, target, reduction="none")[selected] * weights[selected]).sum() / denominator


class InterfacePairs:
    def __init__(self, record: dict[str, Any]) -> None:
        self.record, self.crop_id = record, str(record["crop_id"])
        raw_path = g1.local_path(record["raw_path"])
        if g1.sha256(raw_path) != record["raw_sha256"]:
            raise ValueError(f"{self.crop_id}: immutable raw hash mismatch")
        self.raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        self.targets = np.load(g1.local_path(record["targets"]["path"]), mmap_mode="r", allow_pickle=False)
        self.mask = np.load(g1.local_path(record["mask"]["path"]), mmap_mode="r", allow_pickle=False)
        self.weights = np.load(g1.local_path(record["weights"]["path"]), mmap_mode="r", allow_pickle=False)
        # The receipt is the immutable bridge from target tensor location back
        # to a human decision, raw crop hash, question, and interface ID.
        receipt = json.loads(g1.local_path(record["receipt"]).read_text(encoding="utf-8"))
        if receipt.get("status") != "REVIEWED_DVID_INTERFACE_AFFINITY_SUPERVISION":
            raise ValueError(f"{self.crop_id}: missing reviewed G3 interface receipt")
        if self.targets.shape != (3, *self.raw.shape) or self.mask.shape != self.targets.shape or self.weights.shape != self.targets.shape:
            raise ValueError(f"{self.crop_id}: G3 tensor geometry mismatch")
        self.by_interface: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
        for item in receipt["eligible_pairs"]:
            entry = (int(item["channel_zyx"]), *[int(value) for value in item["pair_left_zyx"]])
            if self.mask[entry] != 1 or self.weights[entry] <= 0:
                raise ValueError(f"{self.crop_id}: manifest-approved pair missing from mask/weights")
            self.by_interface[str(item["interface_id"])].append(entry)
        self.interfaces = sorted(self.by_interface)
        if not self.interfaces:
            raise ValueError(f"{self.crop_id}: no eligible interfaces")
        decisions = {int(self.targets[entry]) for entries in self.by_interface.values() for entry in entries}
        if decisions != {0, 1}:
            raise ValueError(f"{self.crop_id}: requires SAME and DIFFERENT target evidence")
        for interface, entries in self.by_interface.items():
            # This verifies the materializer invariant before GPU execution.
            if not np.isclose(sum(float(self.weights[entry]) for entry in entries), 1.0):
                raise ValueError(f"{self.crop_id}: interface {interface} violates normalized total weight")

    def patch_for(self, entry: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        channel, z, y, x = entry
        origin = tuple(max(0, min(point - extent // 2, limit - extent)) for point, extent, limit in zip((z, y, x), g1.PATCH_ZYX, self.raw.shape, strict=True))
        slices = tuple(slice(start, start + extent) for start, extent in zip(origin, g1.PATCH_ZYX, strict=True))
        raw = np.asarray(self.raw[slices], dtype=np.float32) / 255.0
        target = np.asarray(self.targets[(slice(None),) + slices], dtype=np.float32)
        mask = np.asarray(self.mask[(slice(None),) + slices], dtype=np.float32)
        weights = np.asarray(self.weights[(slice(None),) + slices], dtype=np.float32)
        local = (channel, z - origin[0], y - origin[1], x - origin[2])
        if mask[local] != 1 or weights[local] <= 0:
            raise RuntimeError("G3 target patch lost its anchor")
        return raw, target, mask, weights, {"crop_id": self.crop_id, "anchor_channel_zyx": channel, "anchor_zyx": [z, y, x], "patch_origin_zyx": list(origin), "supervised_pairs_in_patch": int(mask.sum()), "supervised_interface_weight_in_patch": float(weights.sum())}


def metrics(model: torch.nn.Module, pairs: InterfacePairs) -> dict[str, Any]:
    was_training = model.training; model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for interface, entries in pairs.by_interface.items():
            # Evaluation aggregates member predictions back to their one
            # independent reviewed interface instead of inflating pair counts.
            values=[]; truth=None
            for entry in entries:
                raw, targets, mask, weights, metadata = pairs.patch_for(entry)
                affinity, _ = model(torch.from_numpy(raw[None, None]).cuda())
                channel, z, y, x = entry; oz, oy, ox = metadata["patch_origin_zyx"]
                truth = int(targets[channel, z - oz, y - oy, x - ox])
                values.append(float(affinity[0, channel, z - oz, y - oy, x - ox].cpu()))
            rows.append({"interface_id": interface, "target": truth, "mean_affinity": float(np.mean(values)), "pair_affinities": values})
    model.train(was_training)
    same = [row["mean_affinity"] for row in rows if row["target"] == 1]
    different = [row["mean_affinity"] for row in rows if row["target"] == 0]
    if not same or not different:
        raise ValueError(f"{pairs.crop_id}: metrics unexpectedly lack a class")
    q = lambda values: [float(value) for value in np.quantile(values, [0.1, 0.5, 0.9])]
    return {"reviewed_interfaces": len(rows), "same_interface_count": len(same), "different_interface_count": len(different), "same_mean_affinity": float(np.mean(same)), "same_quantiles_p10_p50_p90": q(same), "different_mean_affinity": float(np.mean(different)), "different_quantiles_p10_p50_p90": q(different), "same_accuracy_at_0_5": float(np.mean(np.asarray(same) >= .5)), "different_accuracy_at_0_5": float(np.mean(np.asarray(different) < .5)), "margin": float(np.mean(same) - np.mean(different)), "interface_predictions": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--target-loss-weight", type=float, default=.25)
    parser.add_argument("--resume-checkpoint", type=pathlib.Path, default=ROOT / "experiments/phase6c/MV-TRAIN-SEGNEURON-DVID-G1-001-ATTEMPT-002/checkpoint-step2280.pt")
    args = parser.parse_args()
    if args.steps <= 0 or args.checkpoint_every <= 0 or not 0 < args.target_loss_weight <= 1:
        raise ValueError("Invalid bounded G3 schedule")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if dataset.get("status") != "FROZEN_READY_FOR_G3_LORO" or dataset.get("protected_regression", {}).get("included"):
        raise ValueError("G3 dataset is not frozen or contaminates the protected regression")
    records = {str(record["crop_id"]): record for record in dataset["regions"]}
    held = f"MV-G3-TRAIN-{args.fold}"
    # The held-out region is never passed to the optimizer.  Each fold begins
    # from the same step-2280 MNet + Adam state, not from another fold.
    fold = next((item for item in dataset["loro_folds"] if item["held_out_region"] == held), None)
    if fold is None or held not in records:
        raise ValueError("Unknown frozen G3 LORO fold")
    train_sets = [InterfacePairs(records[name]) for name in fold["train_with"]]
    held_pairs = InterfacePairs(records[held])
    validation = [InterfacePairs(records[f"MV-G3-VALIDATION-V{index}"]) for index in (1, 2, 3)]
    output = ROOT / "experiments/phase6e" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    model = g1.MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub").cuda().train(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    prior = torch.load(args.resume_checkpoint, map_location="cuda", weights_only=False)
    if not {"model_weights", "optimizer", "step"}.issubset(prior):
        raise ValueError("G1 checkpoint lacks exact model/Adam state")
    model.load_state_dict(prior["model_weights"], strict=True); optimizer.load_state_dict(prior["optimizer"]); start = int(prior["step"])
    source_root = ROOT / "datasets/cache/segneuron/EMNeuron-labeled/labeled/Hemi-brain-fib"; source_loss = g1.WeightedBCE(); frozen_source = g1.source_validation(source_root)
    history: list[dict[str, Any]]=[]; checkpoints: list[dict[str, Any]]=[]; best: dict[str, Any] | None=None; started=time.time()
    for index in range(args.steps):
        global_step=start+index+1; source_raw, source_aff, source_fg, source_meta=g1.source_batch(source_root,start+index)
        # Cycle over regions and then interfaces, not raw pair counts.  This
        # complements the per-interface tensor weights and prevents dense
        # interfaces from receiving disproportionate update opportunities.
        target_set=train_sets[index % len(train_sets)]; interface=target_set.interfaces[(index // len(train_sets)) % len(target_set.interfaces)]; entry=target_set.by_interface[interface][0]
        raw,target,mask,weights,target_meta=target_set.patch_for(entry)
        source_aff_pred,source_fg_pred=model(source_raw); target_pred,_=model(torch.from_numpy(raw[None,None]).cuda())
        source_value=source_loss(source_aff_pred,source_aff)+source_loss(source_fg_pred,source_fg)
        target_value=weighted_bce(target_pred,torch.from_numpy(target[None]).cuda(),torch.from_numpy(mask[None]).cuda(),torch.from_numpy(weights[None]).cuda())
        total=source_value+args.target_loss_weight*target_value; optimizer.zero_grad(); total.backward(); optimizer.step(); torch.cuda.synchronize()
        if not np.isfinite(float(total.detach().cpu())): raise RuntimeError(f"Non-finite G3 loss at step {global_step}")
        history.append({"global_step":global_step,"source_loss":float(source_value.detach().cpu()),"target_weighted_affinity_loss":float(target_value.detach().cpu()),"total_loss":float(total.detach().cpu()),"source_batch":source_meta,"target_batch":{**target_meta,"anchor_interface_id":interface},"gpu_memory_allocated_bytes":int(torch.cuda.memory_allocated())})
        if global_step % args.checkpoint_every == 0 or index+1 == args.steps:
            checkpoint=output/f"checkpoint-step{global_step}.pt"; torch.save({"model_weights":model.state_dict(),"optimizer":optimizer.state_dict(),"step":global_step,"parent_checkpoint":str(args.resume_checkpoint.resolve()),"training_classification":"EXPERIMENTAL_G3_INTERFACE_NORMALIZED_LORO","fold":args.fold},checkpoint)
            model.eval()
            with torch.no_grad():
                sa,sf=model(frozen_source[0]); source_val=float((source_loss(sa,frozen_source[1])+source_loss(sf,frozen_source[2])).cpu())
            model.train(); held_metrics=metrics(model,held_pairs); validation_metrics={item.crop_id:metrics(model,item) for item in validation}
            record={"global_step":global_step,"checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":sha256(checkpoint),"source_validation_loss":source_val,"held_out_region":held_metrics,"independent_validation":validation_metrics}; checkpoints.append(record)
            # Reconstruction is deliberately absent from selection.  The
            # bounded LORO prerequisite is held-out reviewed-interface margin.
            if best is None or held_metrics["margin"] > best["held_out_region"]["margin"]: best=record
            (output/"progress.json").write_text(json.dumps({"status":"RUNNING","history":history,"checkpoints":checkpoints,"best_held_out_interface_checkpoint":best,"protected_regression_used":False},indent=2)+"\n",encoding="utf-8")
    receipt={"status":"COMPLETED","classification":"EXPERIMENTAL_G3_INTERFACE_NORMALIZED_LORO","run_id":args.run_id,"fold":args.fold,"held_out_region":held,"train_regions":fold["train_with"],"start_global_step":start,"final_global_step":start+args.steps,"resume":{"checkpoint":str(args.resume_checkpoint.resolve()),"sha256":sha256(args.resume_checkpoint),"exact_model_and_adam_state_restored":True,"scheduler":"NONE"},"dataset":{"path":str(args.dataset.resolve()),"sha256":sha256(args.dataset)},"loss":{"source":"upstream_weighted_bce","target":"reviewed_masked_affinity_weighted_by_normalized_interface","target_loss_weight":args.target_loss_weight},"history":history,"checkpoints":checkpoints,"best_held_out_interface_checkpoint":best,"device":torch.cuda.get_device_name(0),"torch_version":torch.__version__,"torch_hip_version":torch.version.hip,"peak_gpu_memory_allocated_bytes":int(torch.cuda.max_memory_allocated()),"runtime_seconds":time.time()-started,"protected_regression_used":False,"biological_promotions":{"MV-FRAG":0,"MV-N":0,"MV-SYN":0,"MV-CONN":0}}
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
