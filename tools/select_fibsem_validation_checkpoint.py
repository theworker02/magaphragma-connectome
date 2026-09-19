"""Select a SegNeuron checkpoint solely from the frozen FIB-SEM validation pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    source_root = REPOSITORY / "datasets/cache/segneuron/EMNeuron-labeled/labeled/Hemi-brain-fib"
    raw = imageio.volread(source_root / "4.tif")[:20, :128, :128].astype("float32") / 255.0
    label = imageio.volread(source_root / "4_MaskIns.tif")[:20, :128, :128]
    raw_tensor = torch.from_numpy(raw[None, None]).cuda()
    affinity_tensor = torch.from_numpy(seg_to_affgraph(label, mknhood3d(1), pad="replicate").astype("float32")[None]).cuda()
    foreground_tensor = torch.from_numpy((label != 0).astype("float32")[None, None]).cuda()
    loss_function = WeightedBCE()
    candidates = []
    for checkpoint in sorted(args.run_dir.glob("checkpoint-step*.pt"), key=lambda path: int(path.stem.split("step")[-1])):
        model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub").cuda().eval()
        payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
        model.load_state_dict(payload["model_weights"])
        with torch.no_grad():
            affinity_prediction, foreground_prediction = model(raw_tensor)
            loss = loss_function(affinity_prediction, affinity_tensor) + loss_function(foreground_prediction, foreground_tensor)
        candidates.append({"checkpoint": checkpoint.name, "step": int(payload["step"]), "sha256": sha256(checkpoint), "validation_loss": float(loss.detach().cpu())})
        del model
        torch.cuda.empty_cache()
    if not candidates:
        raise ValueError(f"No checkpoints in {args.run_dir}")
    selected = min(candidates, key=lambda candidate: candidate["validation_loss"])
    document = {
        "schema_version": 1,
        "selection_basis": "FROZEN_VALIDATION_ONLY",
        "validation_pair": "Hemi-brain-fib/4",
        "validation_patch_zyx_origin": [0, 0, 0],
        "validation_patch_shape": list(raw.shape),
        "excluded_test_pair": "Hemi-brain-fib/5",
        "candidates": candidates,
        "selected": selected,
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
