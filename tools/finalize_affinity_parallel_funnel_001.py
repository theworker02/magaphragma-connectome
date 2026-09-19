"""Finalize S1 from existing short receipts + extend survivors (post Unicode crash recovery)."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001"
FUNNEL = REPO / "experiments/phase6e/AFFINITY_PARALLEL_QUALIFICATION_FUNNEL_001.json"
MATRIX = REPO / "experiments/phase6e/AFFINITY_PARALLEL_CANDIDATE_MATRIX_001.json"
CLARIFICATION = REPO / "experiments/phase6e/AFFINITY_PARALLEL_EARLY_ELIMINATE_CLARIFICATION_001.json"
WORKER = REPO / "tools/train_affinity_parallel_candidate.py"
WSL_PYTHON = "/opt/venvs/unlearning-rocm/bin/python"
CIDS = ["A_LR_1E3", "B_DIFF_CLASS_WEIGHT_3", "C_PATCH_20_64_64", "D_TARGET_LOSS_WEIGHT_1"]


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def never_left_chance(receipt: dict) -> bool:
    summaries = receipt.get("all_checkpoint_summaries") or []
    if not summaries:
        return True
    for s in summaries:
        bal = float(s["balanced_edge_accuracy_at_0p5"])
        margin = float(s["margin"])
        if abs(bal - 0.5) > 1e-9 or margin > 0.0:
            return False
    return True


def wsl_extend(candidate_id: str) -> dict:
    win = str(WORKER).replace("\\", "/")
    wsl_path = f"/mnt/{win[0].lower()}/{win[3:]}"
    cmd = ["wsl", "-e", "bash", "-lc", f"{WSL_PYTHON} {wsl_path} --candidate {candidate_id} --phase extend"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    receipt_path = OUT_ROOT / candidate_id / "extend" / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
    return {
        "candidate_id": candidate_id,
        "phase": "extend",
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "receipt": receipt,
    }


def main() -> int:
    clarification = json.loads(CLARIFICATION.read_text(encoding="utf-8"))
    s0 = json.loads((OUT_ROOT / "S0_construction_preflight.json").read_text(encoding="utf-8"))

    short_results = []
    survivors = []
    eliminated = []
    for cid in CIDS:
        rec = json.loads((OUT_ROOT / cid / "short" / "receipt.json").read_text(encoding="utf-8"))
        if never_left_chance(rec):
            status = "ELIMINATED_CHANCE"
            eliminated.append({"candidate_id": cid, "status": status, "eval_metrics": rec["eval_metrics"]})
        else:
            status = "SURVIVOR_SHORT"
            survivors.append(cid)
            rec["status"] = status
            rec["verdict"] = f"AFFINITY-PARALLEL-{cid}-SHORT_SURVIVOR_SHORT"
            rec["early_eliminated"] = False
            rec["reclassified_by"] = clarification["id"]
            rec["reclassification_note"] = "Never-left-chance rule; prior ELIMINATED_CHANCE was last-step-only bug"
            (OUT_ROOT / cid / "short" / "receipt.json").write_text(
                json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        short_results.append({"candidate_id": cid, "status": status, "receipt": rec})

    (OUT_ROOT / "S1_short_results.json").write_text(
        json.dumps(
            {
                "created_at": _now(),
                "clarification_id": clarification["id"],
                "short_results": short_results,
                "survivors": survivors,
                "eliminated": eliminated,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"S1 survivors: {survivors}", flush=True)
    print(f"S1 eliminated: {[e['candidate_id'] for e in eliminated]}", flush=True)

    extend_results = []
    passers = []
    if survivors:
        for cid in survivors:
            print(f"S2 extend {cid}...", flush=True)
            extend_results.append(wsl_extend(cid))
            print(f"  finished extend {cid} -> rc={extend_results[-1]['returncode']}", flush=True)
        (OUT_ROOT / "S2_extend_results.json").write_text(
            json.dumps(extend_results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
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
        "funnel_id": "AFFINITY_PARALLEL_QUALIFICATION_FUNNEL_001",
        "matrix_id": "AFFINITY_PARALLEL_CANDIDATE_MATRIX_001",
        "early_eliminate_clarification_id": clarification["id"],
        "S0": s0,
        "S1_eliminated": eliminated,
        "S1_survivors": survivors,
        "S3_passers": passers,
        "next_human_gate": next_human,
        "gate_d_unchanged": True,
        "roi_still_closed": True,
        "full_volume_still_closed": True,
        "dense_segmentation": "STILL_CLOSED",
        "orchestrator_note": "Primary orchestrator crashed after S1 on Windows cp1252 Unicode arrow; shorts had completed; this finalize recovered S1 classification + S2.",
    }
    (OUT_ROOT / "FUNNEL_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    funnel = json.loads(FUNNEL.read_text(encoding="utf-8"))
    if passers:
        funnel["status"] = "EXECUTED_AWAITING_HUMAN_S4"
    elif "FAIL" in status:
        funnel["status"] = "EXECUTED_QUALIFICATION_FAIL"
    funnel["summary_path"] = str((OUT_ROOT / "FUNNEL_SUMMARY.json").relative_to(REPO)).replace("\\", "/")
    funnel["early_eliminate_clarification"] = str(CLARIFICATION.relative_to(REPO)).replace("\\", "/")
    FUNNEL.write_text(json.dumps(funnel, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("status", "S1_survivors", "S3_passers", "next_human_gate")}, indent=2))
    return 0 if passers or survivors else 2


if __name__ == "__main__":
    raise SystemExit(main())
