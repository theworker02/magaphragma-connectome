"""Parameterized short/extend affinity train for AFFINITY_PARALLEL_CANDIDATE_MATRIX_001.

Does not weaken GATE_D. Eval holdout unchanged. Early-eliminate uses frozen rule only.
"""
from __future__ import annotations

import argparse
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

MATRIX = REPO / "experiments/phase6e/AFFINITY_PARALLEL_CANDIDATE_MATRIX_001.json"
FUNNEL = REPO / "experiments/phase6e/AFFINITY_PARALLEL_QUALIFICATION_FUNNEL_001.json"
SUPERVISION = REPO / "experiments/phase6e/AFFINITY-TRAIN-PGT002-001/supervision/supervision_manifest.json"
INIT_CKPT = REPO / "experiments/phase6b/MV-TRAIN-SEGNEURON-FIBSEM-003/checkpoint-step2250.pt"
SOURCE_ROOT = REPO / "datasets/cache/segneuron/EMNeuron-labeled/labeled/Hemi-brain-fib"
OUT_ROOT = REPO / "experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001"


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


def set_seeds(seeds: dict) -> None:
    os.environ["PYTHONHASHSEED"] = str(seeds["python_hash_seed"])
    s = int(seeds["torch_manual_seed"])
    random.seed(s)
    np.random.seed(int(seeds["numpy_seed"]))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_raw(path: str) -> np.ndarray:
    return np.load(win_to_wsl(path), mmap_mode="r", allow_pickle=False)


def patch_for_edge(
    raw: np.ndarray, channel: int, left: list[int], patch_zyx: tuple[int, int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int], list[int]]:
    z, y, x = left
    origin = [
        max(0, min(v - e // 2, lim - e))
        for v, e, lim in zip((z, y, x), patch_zyx, raw.shape, strict=True)
    ]
    slices = tuple(slice(o, o + e) for o, e in zip(origin, patch_zyx, strict=True))
    patch = np.asarray(raw[slices], dtype=np.float32) / 255.0
    local = (channel, z - origin[0], y - origin[1], x - origin[2])
    return patch, local, origin


def source_batch(iteration: int, patch_zyx: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import imageio.v2 as imageio

    pz, py, px = patch_zyx
    source_id = iteration % 4
    root = win_to_wsl(SOURCE_ROOT)
    raw = imageio.volread(root / f"{source_id}.tif")
    label = imageio.volread(root / f"{source_id}_MaskIns.tif")
    z, y, x = (iteration * 13) % 400, (iteration * 29) % 390, (iteration * 37) % 390
    raw_patch = raw[z : z + pz, y : y + py, x : x + px].astype("float32") / 255.0
    label_patch = label[z : z + pz, y : y + py, x : x + px]
    affinity = seg_to_affgraph(label_patch, mknhood3d(1), pad="replicate").astype("float32")
    fg = (label_patch != 0).astype("float32")
    return (
        torch.from_numpy(raw_patch[None, None]).cuda(),
        torch.from_numpy(affinity[None]).cuda(),
        torch.from_numpy(fg[None, None]).cuda(),
    )


def masked_affinity_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    different_multiplier: float,
) -> torch.Tensor:
    selected = mask.bool()
    if not bool(selected.any()):
        raise ValueError("no supervised pairs")
    per = F.binary_cross_entropy(pred, target, reduction="none")
    weights = torch.ones_like(per)
    # DIFFERENT target==0 → upweight; SAME target==1 → weight 1
    weights = torch.where(target < 0.5, torch.full_like(weights, float(different_multiplier)), weights)
    return (per * weights)[selected].mean()


def score_eval_edges(
    model: torch.nn.Module,
    edges: list[dict],
    patch_zyx: tuple[int, int, int],
    threshold: float = 0.5,
) -> dict:
    was = model.training
    model.eval()
    preds: list[tuple[int, float, str]] = []
    with torch.no_grad():
        for e in edges:
            raw = load_raw(e["raw_path"])
            patch, local, _ = patch_for_edge(raw, int(e["channel_zyx"]), list(e["pair_left_zyx"]), patch_zyx)
            aff, _ = model(torch.from_numpy(patch[None, None]).cuda())
            c, lz, ly, lx = (int(local[0]), int(local[1]), int(local[2]), int(local[3]))
            value = aff[0, c, lz, ly, lx].detach().float().cpu().reshape(-1)
            if value.numel() != 1:
                raise RuntimeError(f"expected scalar affinity, got shape {tuple(aff.shape)}")
            preds.append((int(e["target"]), float(value.item()), e["opaque_decision_id"]))
    model.train(was)
    same = [v for t, v, _ in preds if t == 1]
    diff = [v for t, v, _ in preds if t == 0]
    same_acc = float(np.mean(np.asarray(same) >= threshold)) if same else float("nan")
    diff_acc = float(np.mean(np.asarray(diff) < threshold)) if diff else float("nan")
    balanced = float(np.nanmean([same_acc, diff_acc]))
    mean_same = float(np.mean(same)) if same else float("nan")
    mean_diff = float(np.mean(diff)) if diff else float("nan")
    margin = float(mean_same - mean_diff) if same and diff else float("nan")
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


def is_chance(metrics: dict) -> bool:
    bal = float(metrics["balanced_edge_accuracy_at_0p5"])
    margin = float(metrics["margin"])
    return abs(bal - 0.5) <= 1e-9 and margin <= 0.0


def run_candidate(candidate_id: str, phase: str, max_steps: int | None = None) -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm required")
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    funnel = json.loads(FUNNEL.read_text(encoding="utf-8"))
    cand = next((c for c in matrix["candidates"] if c["candidate_id"] == candidate_id), None)
    if cand is None:
        raise SystemExit(f"unknown candidate {candidate_id}")

    cfg = cand["config"]
    patch_zyx = tuple(int(x) for x in cfg["patch_shape_zyx"])
    lr = float(cfg["learning_rate"])
    target_w = float(cfg["target_loss_weight"])
    diff_mult = float(cfg["different_class_loss_multiplier"])
    shared = matrix["shared_frozen_context"]
    seeds = shared["seeds"]
    threshold = float(shared["decision_threshold"])

    short_steps = int(matrix["short_budget"]["steps"])
    extend_steps = int(matrix["extend_budget"]["steps"])
    ckpt_every = int(matrix["short_budget"]["checkpoint_every"])
    eliminate_steps = set(int(x) for x in matrix["early_eliminate"]["at_steps"])

    if phase == "short":
        steps = short_steps if max_steps is None else min(short_steps, int(max_steps))
    elif phase == "extend":
        steps = extend_steps if max_steps is None else min(extend_steps, int(max_steps))
    else:
        raise SystemExit("phase must be short|extend")

    sup = json.loads(SUPERVISION.read_text(encoding="utf-8"))
    if {e["source_id"] for e in sup["train_edges"]} & {e["source_id"] for e in sup["eval_edges"]}:
        raise SystemExit("GATE_E hard fail: train/eval source overlap")

    run_id = f"AFFINITY-PARALLEL-{candidate_id}-{phase.upper()}"
    out = OUT_ROOT / candidate_id / phase
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir()

    set_seeds(seeds)
    init_ckpt = win_to_wsl(INIT_CKPT)
    init_sha = sha256(init_ckpt)

    kn = tuple(shared["mnet_kwargs"]["kn"])
    model = MNet(1, kn=kn, FMU=shared["mnet_kwargs"]["FMU"]).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    prior = torch.load(init_ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(prior["model_weights"], strict=True)
    source_loss_fn = WeightedBCE()

    train_edge_list = sup["train_edges"]
    eval_edge_list = sorted(sup["eval_edges"], key=lambda e: e["opaque_decision_id"])
    history = []
    checkpoints = []
    best = None
    train_sources_seen: set[str] = set()
    eliminated = False
    eliminate_at = None
    started = time.time()

    for index in range(steps):
        global_step = index + 1
        edge = train_edge_list[index % len(train_edge_list)]
        train_sources_seen.add(edge["source_id"])
        if edge["source_id"] in set(sup["eval_sources"]):
            raise SystemExit("eval source leaked into training batch")

        src_raw, src_aff, src_fg = source_batch(index, patch_zyx)
        raw = load_raw(edge["raw_path"])
        patch, local, _ = patch_for_edge(raw, int(edge["channel_zyx"]), list(edge["pair_left_zyx"]), patch_zyx)
        target = np.zeros((3, *patch_zyx), dtype=np.float32)
        mask = np.zeros((3, *patch_zyx), dtype=np.float32)
        target[local] = float(edge["target"])
        mask[local] = 1.0

        src_pred_aff, src_pred_fg = model(src_raw)
        tgt_pred_aff, _ = model(torch.from_numpy(patch[None, None]).cuda())
        source_value = source_loss_fn(src_pred_aff, src_aff) + source_loss_fn(src_pred_fg, src_fg)
        target_value = masked_affinity_bce(
            tgt_pred_aff,
            torch.from_numpy(target[None]).cuda(),
            torch.from_numpy(mask[None]).cuda(),
            different_multiplier=diff_mult,
        )
        total = source_value + target_w * target_value
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

        if global_step % ckpt_every == 0 or global_step == steps:
            ckpt_path = ckpt_dir / f"checkpoint-step{global_step}.pt"
            torch.save(
                {
                    "model_weights": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "matrix_id": matrix["id"],
                    "funnel_id": funnel["id"],
                },
                ckpt_path,
            )
            metrics = score_eval_edges(model, eval_edge_list, patch_zyx, threshold=threshold)
            record = {
                "global_step": global_step,
                "checkpoint": str(ckpt_path).replace("\\", "/"),
                "checkpoint_sha256": sha256(ckpt_path),
                "eval_metrics": metrics,
                "chance_under_early_eliminate_rule": is_chance(metrics),
            }
            checkpoints.append(record)
            bal = float(metrics["balanced_edge_accuracy_at_0p5"])
            if best is None or bal > float(best["eval_metrics"]["balanced_edge_accuracy_at_0p5"]) or (
                bal == float(best["eval_metrics"]["balanced_edge_accuracy_at_0p5"])
                and global_step < int(best["global_step"])
            ):
                best = record

            (out / "progress.json").write_text(
                json.dumps({"status": "RUNNING", "best": best, "last": record}, indent=2) + "\n",
                encoding="utf-8",
            )

            if phase == "short" and global_step in eliminate_steps and is_chance(metrics):
                # Eliminate only if never left chance (best-so-far also chance).
                # Clarification: AFFINITY_PARALLEL_EARLY_ELIMINATE_CLARIFICATION_001
                best_chance = best is None or is_chance(best["eval_metrics"])
                if best_chance and is_chance(metrics):
                    eliminated = True
                    eliminate_at = global_step
                    break

    assert best is not None
    final_metrics = best["eval_metrics"]
    expected_eval_ids = sorted(e["opaque_decision_id"] for e in eval_edge_list)
    scored_ids = sorted(final_metrics["opaque_ids_scored"])
    gate_a = scored_ids == expected_eval_ids and len(scored_ids) == len(eval_edge_list)
    gate_b = any(e["decision"] == "SAME_PROCESS" for e in eval_edge_list) and any(
        e["decision"] == "DIFFERENT_PROCESS" for e in eval_edge_list
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

    if eliminated:
        status = "ELIMINATED_CHANCE"
        verdict = f"{run_id}_ELIMINATED"
    elif phase == "short":
        status = "SURVIVOR_SHORT"
        verdict = f"{run_id}_SURVIVOR_SHORT"
    else:
        all_pass = all(gates.values())
        status = "PASS" if all_pass else "FAIL"
        verdict = f"{run_id}_PASS" if all_pass else f"{run_id}_FAIL"

    receipt = {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "phase": phase,
        "matrix_id": matrix["id"],
        "funnel_id": funnel["id"],
        "created_at": _now(),
        "status": status,
        "verdict": verdict,
        "config": cfg,
        "changes_exactly_one_thing": cand["changes_exactly_one_thing"],
        "tests_failure_class": cand["tests_failure_class"],
        "promoted_gt_sha256": sup["promoted_artifact_sha256"],
        "train_sources": sorted(sup["train_sources"]),
        "eval_sources": sorted(sup["eval_sources"]),
        "train_sources_seen_in_batches": sorted(train_sources_seen),
        "init_checkpoint_path_and_sha256": {
            "path": str(INIT_CKPT).replace("\\", "/"),
            "sha256": init_sha,
        },
        "seeds": seeds,
        "steps_requested": steps,
        "steps_completed": int(history[-1]["global_step"]) if history else 0,
        "checkpoint_every": ckpt_every,
        "early_eliminated": eliminated,
        "early_eliminate_at_step": eliminate_at,
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
        "all_checkpoint_summaries": [
            {
                "global_step": c["global_step"],
                "balanced_edge_accuracy_at_0p5": c["eval_metrics"]["balanced_edge_accuracy_at_0p5"],
                "margin": c["eval_metrics"]["margin"],
                "chance_under_early_eliminate_rule": c["chance_under_early_eliminate_rule"],
            }
            for c in checkpoints
        ],
        "gate_outcomes": gates,
        "do_not_weaken_GATE_D": True,
        "do_not_run_dense_segmentation_from_short_or_fail": True,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "runtime_seconds": time.time() - started,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "status": status, "gates": gates, "eval_metrics": receipt["eval_metrics"]}, indent=2))
    if eliminated:
        return 3
    if status == "FAIL":
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--phase", choices=["short", "extend"], required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    return run_candidate(args.candidate, args.phase, args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())
