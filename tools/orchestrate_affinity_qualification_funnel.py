"""Orchestrator for AFFINITY_PARALLEL_QUALIFICATION_FUNNEL_001.

Default is dry-run (plan only). Pass --execute to run Stage 0 + parallel short candidates.
Does not open ROI/full-volume. Does not weaken GATE_D.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from train_affinity_pgt002_001 import (  # noqa: E402
    INIT_CKPT,
    PATCH_ZYX,
    load_raw,
    patch_for_edge,
    sha256,
    win_to_wsl,
)

FUNNEL = REPO / "experiments/phase6e/AFFINITY_PARALLEL_QUALIFICATION_FUNNEL_001.json"
MATRIX = REPO / "experiments/phase6e/AFFINITY_PARALLEL_CANDIDATE_MATRIX_001.json"
ROI = REPO / "experiments/phase6e/AFFINITY_ROI_RECONSTRUCTION_GATE_001.json"
SUPERVISION = REPO / "experiments/phase6e/AFFINITY-TRAIN-PGT002-001/supervision/supervision_manifest.json"
OUT_ROOT = REPO / "experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001"
WORKER = REPO / "tools/train_affinity_parallel_candidate.py"
WSL_PYTHON = "/opt/venvs/unlearning-rocm/bin/python"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def unit_step(channel: int) -> list[int]:
    e = [0, 0, 0]
    e[channel] = 1
    return e


def construction_preflight(edges: list[dict]) -> dict:
    issues = []
    for e in edges:
        left = list(e["pair_left_zyx"])
        right = list(e["pair_right_zyx"])
        ch = int(e["channel_zyx"])
        expected_right = [a + b for a, b in zip(left, unit_step(ch))]
        if right != expected_right:
            issues.append({"id": e["opaque_decision_id"], "issue": "pair_right_not_plus_e_channel"})
            continue
        raw = load_raw(e["raw_path"])
        if list(raw.shape) != list(e["shape_zyx"]):
            issues.append({"id": e["opaque_decision_id"], "issue": "shape_mismatch"})
            continue
        if sha256(win_to_wsl(e["raw_path"])) != e["raw_sha256"]:
            issues.append({"id": e["opaque_decision_id"], "issue": "raw_sha256_mismatch"})
            continue
        # audit against baseline patch; candidate C uses smaller patch but same left voxel
        patch, local, origin = patch_for_edge(raw, ch, left)
        if patch.shape != PATCH_ZYX:
            issues.append({"id": e["opaque_decision_id"], "issue": "patch_shape", "got": list(patch.shape)})
            continue
        c, lz, ly, lx = local
        abs_zyx = [origin[0] + lz, origin[1] + ly, origin[2] + lx]
        if abs_zyx != left:
            issues.append({"id": e["opaque_decision_id"], "issue": "origin_local_not_left"})
            continue
        if (e["decision"] == "SAME_PROCESS") != (int(e["target"]) == 1):
            issues.append({"id": e["opaque_decision_id"], "issue": "target_decision_mismatch"})
    outcome = "CONSTRUCTION_OK" if not issues else "CONSTRUCTION_DEFECT"
    return {
        "stage_id": "S0_CONSTRUCTION_PREFLIGHT",
        "outcome": outcome,
        "n_edges_audited": len(edges),
        "n_issues": len(issues),
        "issues": issues[:50],
        "supports_F3": outcome == "CONSTRUCTION_DEFECT",
        "created_at": _now(),
    }


def wsl_worker_cmd(candidate_id: str, phase: str) -> list[str]:
    win = str(WORKER).replace("\\", "/")
    # C:/Users/... -> /mnt/c/Users/...
    if len(win) >= 2 and win[1] == ":":
        wsl_path = f"/mnt/{win[0].lower()}/{win[3:]}"
    else:
        wsl_path = win
    return [
        "wsl",
        "-e",
        "bash",
        "-lc",
        f"{WSL_PYTHON} {wsl_path} --candidate {candidate_id} --phase {phase}",
    ]


def run_worker(candidate_id: str, phase: str) -> dict:
    cmd = wsl_worker_cmd(candidate_id, phase)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    receipt_path = OUT_ROOT / candidate_id / phase / "receipt.json"
    receipt = None
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {
        "candidate_id": candidate_id,
        "phase": phase,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "receipt": receipt,
    }


def plan(matrix: dict, funnel: dict) -> dict:
    return {
        "funnel_id": funnel["id"],
        "matrix_id": matrix["id"],
        "stages": [
            "S0_CONSTRUCTION_PREFLIGHT",
            "S1_PARALLEL_SHORT_QUALIFICATION (A–D, 30 steps, early-eliminate)",
            "S2_EXTEND_SURVIVORS_ONLY (to 120)",
            "S3_FROZEN_GATE_A_E",
            "S4_HUMAN_SCIENTIFIC_GATE (manual)",
            "S5_ROI (CLOSED until S4)",
            "S6–S8 CLOSED",
        ],
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "one_factor": c["changes_exactly_one_thing"],
                "tests": c["tests_failure_class"],
                "config": c["config"],
            }
            for c in matrix["candidates"]
        ],
        "early_eliminate": matrix["early_eliminate"],
        "gates_unchanged": True,
        "roi_status": json.loads(ROI.read_text(encoding="utf-8"))["status"],
        "execute": False,
    }


def execute_funnel(parallel: int) -> dict:
    funnel = json.loads(FUNNEL.read_text(encoding="utf-8"))
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    if OUT_ROOT.exists():
        raise FileExistsError(f"Refusing overwrite: {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True)

    sup = json.loads(SUPERVISION.read_text(encoding="utf-8"))
    all_edges = list(sup["train_edges"]) + list(sup["eval_edges"])

    print("S0 construction preflight...", flush=True)
    s0 = construction_preflight(all_edges)
    (OUT_ROOT / "S0_construction_preflight.json").write_text(json.dumps(s0, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if s0["outcome"] != "CONSTRUCTION_OK":
        summary = {
            "id": "AFFINITY_PARALLEL_FUNNEL_001_SUMMARY",
            "created_at": _now(),
            "status": "STOPPED_AFTER_S0_CONSTRUCTION_DEFECT",
            "S0": s0,
            "gate_d_unchanged": True,
            "roi_still_closed": True,
        }
        (OUT_ROOT / "FUNNEL_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    print("S1 parallel short qualification...", flush=True)
    short_results = []
    cids = [c["candidate_id"] for c in matrix["candidates"]]
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futs = {ex.submit(run_worker, cid, "short"): cid for cid in cids}
        for fut in as_completed(futs):
            short_results.append(fut.result())
            print(f"  finished {futs[fut]} -> rc={short_results[-1]['returncode']}", flush=True)

    (OUT_ROOT / "S1_short_results.json").write_text(json.dumps(short_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    survivors = []
    eliminated = []
    for r in short_results:
        rec = r.get("receipt") or {}
        st = rec.get("status")
        if st == "SURVIVOR_SHORT":
            survivors.append(r["candidate_id"])
        else:
            eliminated.append({"candidate_id": r["candidate_id"], "status": st or "NO_RECEIPT", "returncode": r["returncode"]})

    extend_results = []
    if survivors:
        print(f"S2 extend survivors: {survivors}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, min(parallel, len(survivors)))) as ex:
            futs = {ex.submit(run_worker, cid, "extend"): cid for cid in survivors}
            for fut in as_completed(futs):
                extend_results.append(fut.result())
                print(f"  finished extend {futs[fut]} -> rc={extend_results[-1]['returncode']}", flush=True)
        (OUT_ROOT / "S2_extend_results.json").write_text(
            json.dumps(extend_results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        print("S2 skipped — zero short survivors", flush=True)

    passers = []
    for r in extend_results:
        rec = r.get("receipt") or {}
        if rec.get("status") == "PASS" and all((rec.get("gate_outcomes") or {}).values()):
            passers.append(
                {
                    "candidate_id": r["candidate_id"],
                    "selected_checkpoint_step": rec.get("selected_checkpoint_step"),
                    "selected_checkpoint_sha256": rec.get("selected_checkpoint_sha256"),
                    "eval_metrics": rec.get("eval_metrics"),
                    "gate_outcomes": rec.get("gate_outcomes"),
                }
            )

    if passers:
        status = "AFFINITY_PASS_AWAITING_HUMAN_S4"
        next_human = "S4_HUMAN_SCIENTIFIC_GATE_AFFINITY_QUALIFIED — authorize ROI package open"
    elif not survivors:
        status = "QUALIFICATION_FAIL_ZERO_SHORT_SURVIVORS"
        next_human = "Do not invent post-hoc candidates inside this freeze; new matrix under new freeze if needed"
    else:
        status = "QUALIFICATION_FAIL_NO_EXTEND_PASS"
        next_human = "Archive; optional new frozen matrix citing failure classes — GATE_D unchanged"

    summary = {
        "id": "AFFINITY_PARALLEL_FUNNEL_001_SUMMARY",
        "created_at": _now(),
        "status": status,
        "funnel_id": funnel["id"],
        "matrix_id": matrix["id"],
        "S0": s0,
        "S1_eliminated": eliminated,
        "S1_survivors": survivors,
        "S3_passers": passers,
        "next_human_gate": next_human,
        "gate_d_unchanged": True,
        "roi_still_closed": True,
        "full_volume_still_closed": True,
        "dense_segmentation": "STILL_CLOSED",
    }
    (OUT_ROOT / "FUNNEL_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # stamp funnel status
    funnel["status"] = "EXECUTED_SUMMARY_RECORDED" if "FAIL" in status or status.endswith("S4") else funnel["status"]
    if "FAIL" in status:
        funnel["status"] = "EXECUTED_QUALIFICATION_FAIL"
    elif passers:
        funnel["status"] = "EXECUTED_AWAITING_HUMAN_S4"
    funnel["summary_path"] = str((OUT_ROOT / "FUNNEL_SUMMARY.json").relative_to(REPO)).replace("\\", "/")
    FUNNEL.write_text(json.dumps(funnel, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="Actually run S0+S1(+S2). Default: dry-run plan only.")
    ap.add_argument("--parallel", type=int, default=1, help="Max concurrent WSL train workers (GPU memory limited; default 1).")
    args = ap.parse_args()

    funnel = json.loads(FUNNEL.read_text(encoding="utf-8"))
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))

    if not args.execute:
        p = plan(matrix, funnel)
        print(json.dumps(p, indent=2))
        print("\nDry-run only. Re-run with --execute to launch Stage 0 + parallel short qualification.", flush=True)
        return 0

    summary = execute_funnel(parallel=args.parallel)
    print(json.dumps({k: summary[k] for k in ("status", "S1_survivors", "S3_passers", "next_human_gate", "roi_still_closed")}, indent=2))
    if "FAIL" in summary["status"] and "AWAITING" not in summary["status"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
