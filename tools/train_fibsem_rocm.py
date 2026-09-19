"""Run a provenance-recorded FIB-SEM-only supervised SegNeuron training job."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import pathlib
import sys
import time

import imageio
import numpy as np
import torch

REPOSITORY = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(REPOSITORY / "third_party/segneuron/Train_and_Inference"))
from loss.loss import WeightedBCE
from model.Mnet import MNet
from utils.aff_util import seg_to_affgraph
from utils.seg_util import mknhood3d


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def load_source_pair(root_text: str, source_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache only frozen TRAIN source arrays; source files remain untouched."""
    root = pathlib.Path(root_text)
    return imageio.volread(root / f"{source_id}.tif"), imageio.volread(root / f"{source_id}_MaskIns.tif")


def sample(root: pathlib.Path, iteration: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    source_id = iteration % 4  # Frozen TRAIN sources only; Hemi 4 validation and Hemi 5 test excluded.
    raw, label = load_source_pair(str(root), source_id)
    z, y, x = (iteration * 13) % 400, (iteration * 29) % 390, (iteration * 37) % 390
    raw_patch = raw[z : z + 20, y : y + 128, x : x + 128].astype("float32") / 255.0
    label_patch = label[z : z + 20, y : y + 128, x : x + 128]
    affinity = seg_to_affgraph(label_patch, mknhood3d(1), pad="replicate").astype("float32")
    metadata = {
        "source_pair": f"Hemi-brain-fib/{source_id}",
        "patch_zyx_origin": [z, y, x],
        "raw_shape": list(raw_patch.shape),
        "label_shape": list(label_patch.shape),
        "instance_count_in_patch": int(np.unique(label_patch).size),
    }
    return (
        torch.from_numpy(raw_patch[None, None]).cuda(),
        torch.from_numpy(affinity[None]).cuda(),
        torch.from_numpy((label_patch != 0).astype("float32")[None, None]).cuda(),
        metadata,
    )


def validation_batch(root: pathlib.Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The frozen Hemi-brain-fib/4 crop used for checkpoint selection."""
    raw = imageio.volread(root / "4.tif")[:20, :128, :128].astype("float32") / 255.0
    label = imageio.volread(root / "4_MaskIns.tif")[:20, :128, :128]
    affinity = seg_to_affgraph(label, mknhood3d(1), pad="replicate").astype("float32")
    return (
        torch.from_numpy(raw[None, None]).cuda(),
        torch.from_numpy(affinity[None]).cuda(),
        torch.from_numpy((label != 0).astype("float32")[None, None]).cuda(),
    )


def validation_loss(model: torch.nn.Module, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], loss_function: torch.nn.Module) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        raw, affinity, foreground = batch
        affinity_prediction, foreground_prediction = model(raw)
        loss = loss_function(affinity_prediction, affinity) + loss_function(foreground_prediction, foreground)
    model.train(was_training)
    return float(loss.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume-checkpoint", type=pathlib.Path)
    parser.add_argument("--target-regression-every", type=int, default=1000)
    args = parser.parse_args()
    if args.steps <= 0 or args.checkpoint_every <= 0 or args.target_regression_every <= 0:
        raise ValueError("steps and intervals must be positive")
    root = REPOSITORY / "datasets/cache/segneuron/EMNeuron-labeled/labeled/Hemi-brain-fib"
    output = REPOSITORY / "experiments/phase6b" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    cohort_manifest = REPOSITORY / "experiments/phase6b/FIBSEM-supervised-cohort.json"
    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub").cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    start_step = 0
    resume = {"exact_optimizer_state_restored": False, "scheduler": "NONE"}
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume_checkpoint}")
        prior = torch.load(args.resume_checkpoint, map_location="cuda", weights_only=False)
        if "model_weights" not in prior or "optimizer" not in prior or "step" not in prior:
            raise ValueError("Resume checkpoint must contain model_weights, optimizer, and step")
        model.load_state_dict(prior["model_weights"], strict=True)
        optimizer.load_state_dict(prior["optimizer"])
        start_step = int(prior["step"])
        resume = {"checkpoint": str(args.resume_checkpoint.resolve()), "checkpoint_sha256": sha256(args.resume_checkpoint),
                  "source_global_step": start_step, "exact_optimizer_state_restored": True, "scheduler": "NONE"}
    loss_function = WeightedBCE()
    frozen_validation = validation_batch(root)
    losses: list[float] = []
    checkpoints: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    started = time.time()
    for index in range(args.steps):
        raw, affinity, foreground, metadata = sample(root, start_step + index)
        affinity_prediction, foreground_prediction = model(raw)
        loss = loss_function(affinity_prediction, affinity) + loss_function(foreground_prediction, foreground)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        loss_value = float(loss.detach().cpu())
        if not np.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss at global step {start_step + index + 1}: {loss_value}")
        losses.append(loss_value)
        step = start_step + index + 1
        if step % args.checkpoint_every == 0 or step == start_step + args.steps:
            checkpoint_path = output / f"checkpoint-step{step}.pt"
            torch.save({"model_weights": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, checkpoint_path)
            checkpoints.append({"step": step, "path": str(checkpoint_path), "sha256": sha256(checkpoint_path)})
            validation = validation_loss(model, frozen_validation, loss_function)
            if not np.isfinite(validation):
                raise RuntimeError(f"Non-finite frozen validation loss at global step {step}: {validation}")
            validations.append({"step": step, "validation_pair": "Hemi-brain-fib/4", "validation_loss": validation})
            (output / "progress.json").write_text(json.dumps({
                "status": "RUNNING", "step": step, "loss": loss_value, "losses": losses,
                "last_batch": metadata, "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
                "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved()), "checkpoints": checkpoints,
                "frozen_validation": validations, "target_regression_due": step % args.target_regression_every == 0,
                "target_regression_schedule": {"every_steps": args.target_regression_every, "input": "MV-GTVOL-000004", "pipeline": "frozen true-3-D watershed -> RAG -> edge evidence"},
            }, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "status": "COMPLETED", "run_id": args.run_id, "classification": "SOURCE_SUPERVISED_TRAINING",
        "implementation": "UPSTREAM_SEGNEURON_MNet_WITH_FIBSEM_ONLY_COHORT_ADAPTER",
        "invocation_steps": args.steps, "start_global_step": start_step, "final_global_step": start_step + args.steps,
        "losses": losses, "checkpoint_every": args.checkpoint_every, "checkpoints": checkpoints, "frozen_validation": validations,
        "target_regression_every": args.target_regression_every, "resume": resume,
        "source_cohort_manifest": str(cohort_manifest), "source_cohort_manifest_sha256": sha256(cohort_manifest),
        "train_source_pairs": ["Hemi-brain-fib/0", "Hemi-brain-fib/1", "Hemi-brain-fib/2", "Hemi-brain-fib/3"],
        "excluded_validation_source_pair": "Hemi-brain-fib/4", "excluded_test_source_pair": "Hemi-brain-fib/5",
        "optimizer": "Adam", "learning_rate": 0.0001, "scheduler": "NONE", "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__, "torch_hip_version": torch.version.hip,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()), "runtime_seconds": time.time() - started,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
