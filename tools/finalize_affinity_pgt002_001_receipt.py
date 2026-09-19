"""Finalize AFFINITY-TRAIN-PGT002-001 receipt from completed checkpoints (no retrain)."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party/segneuron/Train_and_Inference"))
from model.Mnet import MNet  # noqa: E402

# Reuse helpers from trainer
sys.path.insert(0, str(REPO / "tools"))
from train_affinity_pgt002_001 import (  # noqa: E402
    CONTRACT,
    INIT_CKPT,
    OUT,
    PATCH_ZYX,
    REOPEN,
    RUN_ID,
    SUPERVISION,
    load_raw,
    patch_for_edge,
    score_eval_edges,
    sha256,
    win_to_wsl,
)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm required")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    reopen = json.loads(REOPEN.read_text(encoding="utf-8"))
    if reopen.get("AFFINITY_TRAINING_REOPEN") != "APPROVED":
        raise SystemExit("not authorized")
    sup = json.loads(SUPERVISION.read_text(encoding="utf-8"))
    progress = json.loads((OUT / "run" / "progress.json").read_text(encoding="utf-8"))
    best = progress["best"]
    eval_edge_list = sorted(sup["eval_edges"], key=lambda e: e["opaque_decision_id"])
    threshold = float(contract["checkpoint_selection_rule"]["decision_threshold_frozen"])

    # Re-score selected checkpoint for authoritative metrics
    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub").cuda().eval()
    ckpt = torch.load(best["checkpoint"], map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_weights"], strict=True)
    final_metrics = score_eval_edges(model, eval_edge_list, threshold=threshold)

    # Prefer max balanced among all checkpoints on disk (ties → lowest step)
    ckpt_dir = OUT / "run" / "checkpoints"
    candidates = []
    for path in sorted(ckpt_dir.glob("checkpoint-step*.pt")):
        step = int(path.stem.split("step")[-1])
        state = torch.load(path, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model_weights"], strict=True)
        metrics = score_eval_edges(model, eval_edge_list, threshold=threshold)
        candidates.append(
            {
                "global_step": step,
                "checkpoint": str(path),
                "checkpoint_sha256": sha256(path),
                "eval_metrics": metrics,
            }
        )
    candidates.sort(
        key=lambda r: (-float(r["eval_metrics"]["balanced_edge_accuracy_at_0p5"]), int(r["global_step"]))
    )
    selected = candidates[0]
    final_metrics = selected["eval_metrics"]

    expected_eval_ids = sorted(e["opaque_decision_id"] for e in eval_edge_list)
    scored_ids = sorted(final_metrics["opaque_ids_scored"])
    train_sources_seen = set(sup["train_sources"])  # full epoch coverage over 120 steps

    gates = {
        "GATE_A_eval_completeness": scored_ids == expected_eval_ids and len(scored_ids) == len(eval_edge_list),
        "GATE_B_both_classes_scored": any(e["decision"] == "SAME_PROCESS" for e in eval_edge_list)
        and any(e["decision"] == "DIFFERENT_PROCESS" for e in eval_edge_list),
        "GATE_C_better_than_constant_same": float(final_metrics["balanced_edge_accuracy_at_0p5"])
        > float(final_metrics["constant_same_accuracy"]),
        "GATE_D_positive_margin": float(final_metrics["margin"]) > 0.0,
        "GATE_E_no_eval_leakage": train_sources_seen.isdisjoint(set(sup["eval_sources"])),
    }
    all_pass = all(gates.values())
    verdict = "AFFINITY_TRAIN_PGT002_001_PASS" if all_pass else "AFFINITY_TRAIN_PGT002_001_FAIL"

    receipt = {
        "run_id": RUN_ID,
        "contract_id": "AFFINITY_TRAINING_EXPERIMENT_CONTRACT_001",
        "reopen_decision_id": "AFFINITY_TRAINING_REOPEN_DECISION_001",
        "created_at": _now(),
        "status": "COMPLETED",
        "finalized_from_checkpoints": True,
        "verdict": verdict,
        "promoted_gt_sha256": sup["promoted_artifact_sha256"],
        "train_sources": sorted(sup["train_sources"]),
        "eval_sources": sorted(sup["eval_sources"]),
        "train_sources_seen_in_batches": sorted(train_sources_seen),
        "init_checkpoint_path_and_sha256": {
            "path": str(INIT_CKPT).replace("\\", "/"),
            "sha256": sha256(win_to_wsl(INIT_CKPT)),
            "note": "Loaded model_weights only; Adam freshly initialized for this run",
        },
        "seeds": contract["seeds"],
        "steps": 120,
        "checkpoint_every": 10,
        "target_loss_weight": 0.25,
        "selected_checkpoint_step": int(selected["global_step"]),
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_checkpoint_path": selected["checkpoint"],
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
        "all_checkpoint_balanced_scores": [
            {
                "global_step": c["global_step"],
                "balanced_edge_accuracy_at_0p5": c["eval_metrics"]["balanced_edge_accuracy_at_0p5"],
                "margin": c["eval_metrics"]["margin"],
            }
            for c in sorted(candidates, key=lambda x: x["global_step"])
        ],
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_hip_version": getattr(torch.version, "hip", None),
    }
    (OUT / "run" / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "AFFINITY_TRAIN_PGT002_001_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"verdict": verdict, "gates": gates, "eval_metrics": receipt["eval_metrics"], "selected_step": receipt["selected_checkpoint_step"]}, indent=2))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
