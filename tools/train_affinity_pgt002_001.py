"""First controlled affinity train under AFFINITY_TRAINING_EXPERIMENT_CONTRACT_001.

Consumes only PRODUCTION_GT_PROMOTED_BATCH_002 via the frozen supervision manifest.
Checkpoint selection uses eval balanced_edge_accuracy@0.5 only.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party/segneuron/Train_and_Inference"))
from loss.loss import WeightedBCE  # noqa: E402
from model.Mnet import MNet  # noqa: E402
from utils.aff_util import seg_to_affgraph  # noqa: E402
from utils.seg_util import mknhood3d  # noqa: E402

CONTRACT = REPO / "experiments/phase6e/AFFINITY_TRAINING_EXPERIMENT_CONTRACT_001.json"
REOPEN = REPO / "experiments/phase6e/AFFINITY_TRAINING_REOPEN_DECISION_001.json"
SUPERVISION = REPO / "experiments/phase6e/AFFINITY-TRAIN-PGT002-001/supervision/supervision_manifest.json"
OUT = REPO / "experiments/phase6e/AFFINITY-TRAIN-PGT002-001"
INIT_CKPT = REPO / "experiments/phase6b/MV-TRAIN-SEGNEURON-FIBSEM-003/checkpoint-step2250.pt"
SOURCE_ROOT = REPO / "datasets/cache/segneuron/EMNeuron-labeled/labeled/Hemi-brain-fib"
PATCH_ZYX = (20, 128, 128)
STEPS = 120
CKPT_EVERY = 10
TARGET_LOSS_WEIGHT = 0.25
RUN_ID = "AFFINITY-TRAIN-PGT002-001"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def win_to_wsl(path: Path | str) -> Path:
    """Resolve data paths for the current OS (Windows native vs WSL/Linux)."""
    text = str(path).replace("\\", "/")
    on_linux = sys.platform.startswith("linux")
    if text.startswith("/mnt/") and not on_linux:
        # /mnt/c/Users/... -> C:/Users/...
        parts = text.split("/")
        if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
            drive = parts[2].upper()
            rest = "/".join(parts[3:])
            return Path(f"{drive}:/{rest}")
    if text.startswith("/") and on_linux:
        return Path(text)
    if len(text) >= 2 and text[1] == ":":
        if on_linux:
            drive = text[0].lower()
            rest = text[3:] if len(text) > 2 and text[2] == "/" else text[2:]
            return Path(f"/mnt/{drive}/{rest}")
        return Path(text)
    return Path(text)


def set_seeds(contract: dict) -> None:
    os.environ["PYTHONHASHSEED"] = str(contract["seeds"]["python_hash_seed"])
    s = int(contract["seeds"]["torch_manual_seed"])
    random.seed(s)
    np.random.seed(int(contract["seeds"]["numpy_seed"]))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_raw(path: str) -> np.ndarray:
    p = win_to_wsl(path)
    return np.load(p, mmap_mode="r", allow_pickle=False)


def patch_for_edge(raw: np.ndarray, channel: int, left: list[int]) -> tuple[np.ndarray, tuple[int, int, int, int], list[int]]:
    z, y, x = left
    origin = [
        max(0, min(v - e // 2, lim - e))
        for v, e, lim in zip((z, y, x), PATCH_ZYX, raw.shape, strict=True)
    ]
    slices = tuple(slice(o, o + e) for o, e in zip(origin, PATCH_ZYX, strict=True))
    patch = np.asarray(raw[slices], dtype=np.float32) / 255.0
    local = (channel, z - origin[0], y - origin[1], x - origin[2])
    return patch, local, origin


def source_batch(iteration: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import imageio.v2 as imageio

    source_id = iteration % 4
    root = win_to_wsl(SOURCE_ROOT)
    raw = imageio.volread(root / f"{source_id}.tif")
    label = imageio.volread(root / f"{source_id}_MaskIns.tif")
    z, y, x = (iteration * 13) % 400, (iteration * 29) % 390, (iteration * 37) % 390
    raw_patch = raw[z : z + 20, y : y + 128, x : x + 128].astype("float32") / 255.0
    label_patch = label[z : z + 20, y : y + 128, x : x + 128]
    affinity = seg_to_affgraph(label_patch, mknhood3d(1), pad="replicate").astype("float32")
    fg = (label_patch != 0).astype("float32")
    return (
        torch.from_numpy(raw_patch[None, None]).cuda(),
        torch.from_numpy(affinity[None]).cuda(),
        torch.from_numpy(fg[None, None]).cuda(),
    )


def masked_affinity_bce(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = mask.bool()
    if not bool(selected.any()):
        raise ValueError("no supervised pairs")
    return F.binary_cross_entropy(pred, target, reduction="none")[selected].mean()


def score_eval_edges(model: torch.nn.Module, edges: list[dict], threshold: float = 0.5) -> dict:
    was = model.training
    model.eval()
    preds: list[tuple[int, float, str]] = []
    with torch.no_grad():
        for e in edges:
            raw = load_raw(e["raw_path"])
            patch, local, _ = patch_for_edge(raw, int(e["channel_zyx"]), list(e["pair_left_zyx"]))
            aff, _ = model(torch.from_numpy(patch[None, None]).cuda())
            c, lz, ly, lx = (int(local[0]), int(local[1]), int(local[2]), int(local[3]))
            value = aff[0, c, lz, ly, lx].detach().float().cpu().reshape(-1)
            if value.numel() != 1:
                raise RuntimeError(f"expected scalar affinity, got shape {tuple(aff.shape)} local={(c,lz,ly,lx)}")
            value = float(value.item())
            preds.append((int(e["target"]), value, e["opaque_decision_id"]))
    model.train(was)
    same = [v for t, v, _ in preds if t == 1]
    diff = [v for t, v, _ in preds if t == 0]
    same_acc = float(np.mean(np.asarray(same) >= threshold)) if same else float("nan")
    diff_acc = float(np.mean(np.asarray(diff) < threshold)) if diff else float("nan")
    balanced = float(np.nanmean([same_acc, diff_acc]))
    mean_same = float(np.mean(same)) if same else float("nan")
    mean_diff = float(np.mean(diff)) if diff else float("nan")
    margin = float(mean_same - mean_diff) if same and diff else float("nan")
    # constant SAME predictor accuracy on these edges
    const_same = float(np.mean([1.0 if t == 1 else 0.0 for t, _, _ in preds]))
    return {
        "n_eval_edges": len(preds),
        "opaque_ids_scored": [oid for _, _, oid in preds],
        "SAME_PROCESS_accuracy_at_0p5": same_acc,
        "DIFFERENT_PROCESS_accuracy_at_0p5": diff_acc,
        "balanced_edge_accuracy_at_0p5": balanced,
        "mean_pred_SAME": mean_same,
        "mean_pred_DIFFERENT": mean_diff,
        "margin": margin,
        "constant_same_accuracy": const_same,
        "pair_predictions": [{"target": t, "affinity": v, "opaque_decision_id": oid} for t, v, oid in preds],
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm required")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    reopen = json.loads(REOPEN.read_text(encoding="utf-8"))
    if reopen.get("AFFINITY_TRAINING_REOPEN") != "APPROVED":
        raise SystemExit("training not authorized")
    if not SUPERVISION.exists():
        raise SystemExit("run materialize_affinity_pgt002_supervision.py first")
    sup = json.loads(SUPERVISION.read_text(encoding="utf-8"))
    if {e["source_id"] for e in sup["train_edges"]} & {e["source_id"] for e in sup["eval_edges"]}:
        raise SystemExit("GATE_E hard fail: train/eval source overlap")

    run_dir = OUT / "run"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()

    set_seeds(contract)
    init_ckpt = win_to_wsl(INIT_CKPT)
    init_sha = sha256(init_ckpt)

    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub").cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(contract["model_config"]["learning_rate"]))
    prior = torch.load(init_ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(prior["model_weights"], strict=True)
    # Fresh Adam for this controlled run (document); do not inherit FIB-SEM Adam state into PGT selection.
    start_step = 0
    source_loss_fn = WeightedBCE()

    train_edge_list = sup["train_edges"]
    eval_edge_list = sorted(sup["eval_edges"], key=lambda e: e["opaque_decision_id"])
    history = []
    checkpoints = []
    best = None
    threshold = float(contract["checkpoint_selection_rule"]["decision_threshold_frozen"])
    started = time.time()
    train_sources_seen: set[str] = set()

    for index in range(STEPS):
        global_step = start_step + index + 1
        edge = train_edge_list[index % len(train_edge_list)]
        train_sources_seen.add(edge["source_id"])
        if edge["source_id"] in set(sup["eval_sources"]):
            raise SystemExit("eval source leaked into training batch")

        src_raw, src_aff, src_fg = source_batch(index)
        raw = load_raw(edge["raw_path"])
        patch, local, origin = patch_for_edge(raw, int(edge["channel_zyx"]), list(edge["pair_left_zyx"]))
        target = np.zeros((3, *PATCH_ZYX), dtype=np.float32)
        mask = np.zeros((3, *PATCH_ZYX), dtype=np.float32)
        target[local] = float(edge["target"])
        mask[local] = 1.0

        src_pred_aff, src_pred_fg = model(src_raw)
        tgt_pred_aff, _ = model(torch.from_numpy(patch[None, None]).cuda())
        source_value = source_loss_fn(src_pred_aff, src_aff) + source_loss_fn(src_pred_fg, src_fg)
        target_value = masked_affinity_bce(
            tgt_pred_aff,
            torch.from_numpy(target[None]).cuda(),
            torch.from_numpy(mask[None]).cuda(),
        )
        total = source_value + TARGET_LOSS_WEIGHT * target_value
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        torch.cuda.synchronize()

        history.append(
            {
                "global_step": global_step,
                "source_loss": float(source_value.detach().cpu()),
                "target_masked_affinity_bce": float(target_value.detach().cpu()),
                "total_loss": float(total.detach().cpu()),
                "train_opaque_decision_id": edge["opaque_decision_id"],
                "train_source_id": edge["source_id"],
            }
        )

        if global_step % CKPT_EVERY == 0 or index + 1 == STEPS:
            ckpt_path = ckpt_dir / f"checkpoint-step{global_step}.pt"
            torch.save(
                {
                    "model_weights": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "run_id": RUN_ID,
                    "contract_id": "AFFINITY_TRAINING_EXPERIMENT_CONTRACT_001",
                },
                ckpt_path,
            )
            metrics = score_eval_edges(model, eval_edge_list, threshold=threshold)
            record = {
                "global_step": global_step,
                "checkpoint": str(ckpt_path),
                "checkpoint_sha256": sha256(ckpt_path),
                "eval_metrics": metrics,
            }
            checkpoints.append(record)
            bal = float(metrics["balanced_edge_accuracy_at_0p5"])
            if best is None or bal > float(best["eval_metrics"]["balanced_edge_accuracy_at_0p5"]) or (
                bal == float(best["eval_metrics"]["balanced_edge_accuracy_at_0p5"])
                and global_step < int(best["global_step"])
            ):
                best = record
            (run_dir / "progress.json").write_text(
                json.dumps({"status": "RUNNING", "best": best, "checkpoints": checkpoints[-3:]}, indent=2) + "\n",
                encoding="utf-8",
            )

    assert best is not None
    # Reload best for final eval metrics (already stored)
    final_metrics = best["eval_metrics"]
    expected_eval_ids = sorted(e["opaque_decision_id"] for e in eval_edge_list)
    scored_ids = sorted(final_metrics["opaque_ids_scored"])

    gate_a = scored_ids == expected_eval_ids and len(scored_ids) == len(eval_edge_list)
    gate_b = (
        final_metrics["SAME_PROCESS_accuracy_at_0p5"] == final_metrics["SAME_PROCESS_accuracy_at_0p5"]
        and final_metrics["DIFFERENT_PROCESS_accuracy_at_0p5"] == final_metrics["DIFFERENT_PROCESS_accuracy_at_0p5"]
        and any(e["decision"] == "SAME_PROCESS" for e in eval_edges)
        and any(e["decision"] == "DIFFERENT_PROCESS" for e in eval_edges)
    )
    gate_c = float(final_metrics["balanced_edge_accuracy_at_0p5"]) > float(final_metrics["constant_same_accuracy"])
    gate_d = float(final_metrics["margin"]) > 0.0
    gate_e = train_sources_seen.isdisjoint(set(sup["eval_sources"])) and set(sup["train_sources"]).isdisjoint(
        set(sup["eval_sources"])
    )

    gates = {
        "GATE_A_eval_completeness": gate_a,
        "GATE_B_both_classes_scored": gate_b,
        "GATE_C_better_than_constant_same": gate_c,
        "GATE_D_positive_margin": gate_d,
        "GATE_E_no_eval_leakage": gate_e,
    }
    all_pass = all(gates.values())
    verdict = "AFFINITY_TRAIN_PGT002_001_PASS" if all_pass else "AFFINITY_TRAIN_PGT002_001_FAIL"

    receipt = {
        "run_id": RUN_ID,
        "contract_id": "AFFINITY_TRAINING_EXPERIMENT_CONTRACT_001",
        "reopen_decision_id": "AFFINITY_TRAINING_REOPEN_DECISION_001",
        "created_at": _now(),
        "status": "COMPLETED",
        "verdict": verdict,
        "promoted_gt_sha256": sup["promoted_artifact_sha256"],
        "train_sources": sorted(sup["train_sources"]),
        "eval_sources": sorted(sup["eval_sources"]),
        "train_sources_seen_in_batches": sorted(train_sources_seen),
        "init_checkpoint_path_and_sha256": {
            "path": str(INIT_CKPT).replace("\\", "/"),
            "sha256": init_sha,
            "note": "Loaded model_weights only; Adam freshly initialized for this run",
        },
        "seeds": contract["seeds"],
        "steps": STEPS,
        "checkpoint_every": CKPT_EVERY,
        "target_loss_weight": TARGET_LOSS_WEIGHT,
        "selected_checkpoint_step": int(best["global_step"]),
        "selected_checkpoint_sha256": best["checkpoint_sha256"],
        "selected_checkpoint_path": best["checkpoint"],
        "eval_metrics": {
            k: final_metrics[k]
            for k in (
                "n_eval_edges",
                "SAME_PROCESS_accuracy_at_0p5",
                "DIFFERENT_PROCESS_accuracy_at_0p5",
                "balanced_edge_accuracy_at_0p5",
                "mean_pred_SAME",
                "mean_pred_DIFFERENT",
                "margin",
            )
        },
        "eval_metrics_extended": final_metrics,
        "gate_outcomes": gates,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "runtime_seconds": time.time() - started,
        "history_tail": history[-5:],
        "n_checkpoints": len(checkpoints),
    }
    (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # also top-level experiment receipt
    (OUT / "AFFINITY_TRAIN_PGT002_001_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"verdict": verdict, "gates": gates, "eval_metrics": receipt["eval_metrics"], "selected_step": receipt["selected_checkpoint_step"]}, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
