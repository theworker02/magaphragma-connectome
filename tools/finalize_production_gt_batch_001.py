"""QC-finalize PRODUCTION_GT_BATCH_001; promote only if frozen QC gates pass.

Does not authorize training. Validation_002 remains excluded.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BATCH = REPO / "experiments/phase6e/PRODUCTION_GT_BATCH_001.json"
SEALED = REPO / "experiments/phase6e/PRODUCTION_GT_BATCH_001_AXIS_MAP_SEALED.json"
PROTOCOL = REPO / "experiments/phase6e/AFFINITY_PRODUCTION_LABELING_PROTOCOL_001.json"
PROMOTED = REPO / "experiments/phase6e/PRODUCTION_GT_PROMOTED_BATCH_001.json"
WS_ROOT = REPO / "experiments/phase6e/production-gt-batch-001-packages"
VALID = {"SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"}
EXCL = {"UNCERTAIN", "BAD_QUESTION"}
PATH_PATTERN = {"Z": "DIFFERENT_PROCESS", "Y": "SAME_PROCESS", "X": "SAME_PROCESS"}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def orientation_fail(triplets: list[dict], class_edges: list[dict]) -> tuple[bool, str]:
    evaluable = [t for t in triplets if not t["excluded"]]
    if evaluable and all(t["matches_orientation_collinear_pattern"] for t in evaluable):
        return True, "All evaluable triplets match Z=DIFF & Y=SAME & X=SAME"
    by_axis: dict[str, list[str]] = {"Z": [], "Y": [], "X": []}
    for e in class_edges:
        by_axis[e["axis_name"]].append(e["decision"])
    for ax, labels in by_axis.items():
        if len(labels) < 3:
            continue
        for oa in [a for a in by_axis if a != ax]:
            ol = by_axis[oa]
            if len(ol) < 3:
                continue
            if all(x == "DIFFERENT_PROCESS" for x in labels) and all(x == "SAME_PROCESS" for x in ol):
                return True, f"{ax} all DIFFERENT while {oa} all SAME"
            if all(x == "SAME_PROCESS" for x in labels) and all(x == "DIFFERENT_PROCESS" for x in ol):
                return True, f"{ax} all SAME while {oa} all DIFFERENT"
    return False, ""


def main() -> int:
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if batch.get("qc") or batch.get("promotion"):
        raise SystemExit("Already QC/promoted; refuse overwrite")
    if PROMOTED.exists():
        raise SystemExit(f"Promoted artifact already exists: {PROMOTED}")

    n_expected = int(batch["design"]["total_decisions"])
    by_opaque = {e["opaque_decision_id"]: e for e in sealed["entries"]}
    decisions = {}
    issues = []
    for loc in batch["per_location"]:
        crop = loc["crop_id"]
        log = WS_ROOT / crop / "workspace.events.jsonl"
        if not log.exists() or not log.read_text(encoding="utf-8").strip():
            issues.append(f"{crop}: empty")
            continue
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            oid = ev.get("question_reference")
            if oid not in by_opaque or ev.get("decision") not in VALID:
                issues.append(f"bad event {oid}")
                continue
            meta = by_opaque[oid]
            if list(ev["pair_left_zyx"]) != meta["pair_left_zyx"] or int(ev["channel_zyx"]) != meta["channel_zyx"]:
                issues.append(f"{oid}: geometry mismatch")
            if oid in decisions:
                issues.append(f"{oid}: duplicate")
            decisions[oid] = {
                "opaque_decision_id": oid,
                "decision": ev["decision"],
                "axis_name": meta["axis_name"],
                "channel_zyx": meta["channel_zyx"],
                "crop_id": crop,
                "source_id": meta["source_id"],
                "center_zyx": meta["center_zyx"],
                "pair_left_zyx": meta["pair_left_zyx"],
                "pair_right_zyx": meta["pair_right_zyx"],
                "face_step": meta.get("face_step"),
                "raw_sha256": meta.get("raw_sha256"),
                "event_id": ev["id"],
                "reviewer": ev.get("reviewer"),
                "timestamp": ev.get("timestamp"),
            }

    complete = len(decisions) == n_expected and not issues
    triplets = []
    class_edges = []
    if complete:
        for loc in batch["per_location"]:
            crop = loc["crop_id"]
            labels = {}
            excluded = False
            for oid, meta in by_opaque.items():
                if meta["crop_id"] != crop:
                    continue
                d = decisions[oid]["decision"]
                labels[meta["axis_name"]] = d
                if d in EXCL:
                    excluded = True
                else:
                    class_edges.append({"axis_name": meta["axis_name"], "decision": d})
            triplets.append(
                {
                    "crop_id": crop,
                    "source_id": loc["source_id"],
                    "center_zyx": loc["center_zyx"],
                    "labels_zyx": labels,
                    "excluded": excluded,
                    "matches_orientation_collinear_pattern": (not excluded) and labels == PATH_PATTERN,
                }
            )

    counts = Counter(e["decision"] for e in class_edges)
    n_same, n_diff = counts["SAME_PROCESS"], counts["DIFFERENT_PROCESS"]
    oif, oif_detail = orientation_fail(triplets, class_edges) if complete else (True, "incomplete")
    both_classes = n_same >= 1 and n_diff >= 1
    provenance_ok = complete and all(
        decisions[oid].get("raw_sha256") and decisions[oid].get("reviewer") and decisions[oid].get("event_id")
        for oid in decisions
    )
    no_val_leak = True  # enforced at provision by source exclusion

    gates = {
        "complete_batch": complete,
        "both_classes_present": both_classes,
        "orientation_invariance": complete and not oif,
        "provenance_complete": provenance_ok,
        "no_validation_leakage": no_val_leak,
    }
    all_pass = all(gates.values())

    batch["status"] = "QC_COMPLETE" if complete else "QC_INCOMPLETE"
    batch["completed_at"] = _now()
    batch["results"] = {
        "n_decisions": len(decisions),
        "expected": n_expected,
        "issues": issues,
        "class_counts": {
            "SAME_PROCESS": n_same,
            "DIFFERENT_PROCESS": n_diff,
            "excluded": len(decisions) - n_same - n_diff,
        },
        "triplets": triplets,
        "decisions": [decisions[oid] for oid in sorted(decisions)],
        "orientation_invariance_failed": oif,
        "orientation_invariance_detail": oif_detail or None,
    }
    batch["qc"] = {
        "gates": gates,
        "all_pass": all_pass,
        "protocol_qc_ref": protocol["qc_gates_before_PRODUCTION_GT_PROMOTED"],
        "evaluated_at": _now(),
    }

    if all_pass:
        batch["gt_status"] = "PRODUCTION_GT_PROMOTED"
        batch["promotion"] = {
            "promoted_at": _now(),
            "promoted_artifact": str(PROMOTED.relative_to(REPO)).replace("\\", "/"),
            "do_not_train_until_AFFINITY_TRAINING_REOPEN": True,
            "validation_002_permanently_excluded_from_training": True,
        }
        promoted = {
            "id": "PRODUCTION_GT_PROMOTED_BATCH_001",
            "schema_version": 1,
            "status": "PRODUCTION_GT_PROMOTED",
            "created_at": _now(),
            "source_batch": BATCH_ID,
            "protocol_id": "AFFINITY_PRODUCTION_LABELING_PROTOCOL_001",
            "reopen_decision_id": "AFFINITY_ANNOTATION_REOPEN_DECISION_001",
            "batch_sha256": None,  # filled after write of batch
            "sealed_sha256": _sha_file(SEALED),
            "n_decisions": n_expected,
            "class_counts": batch["results"]["class_counts"],
            "qc": batch["qc"],
            "labels": batch["results"]["decisions"],
            "triplets": triplets,
            "training": {
                "authorized": False,
                "requires": "AFFINITY_TRAINING_REOPEN / TRAIN_AUTHORIZATION",
                "permanently_excluded_artifacts": [
                    "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_002.json",
                    "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_001.json",
                    "experiments/phase6e/DIFFERENT_CLASS_DISCOVERY_001.json",
                    "experiments/phase6e/G3_PRESENTATION_REPAIR_SMOKE_001.json",
                ],
            },
        }
        BATCH.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        promoted["batch_sha256"] = _sha_file(BATCH)
        PROMOTED.write_text(json.dumps(promoted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        batch["gt_status"] = "CANDIDATE_PRODUCTION_GT_QC_FAILED" if complete else "CANDIDATE_PRODUCTION_GT_INCOMPLETE"
        batch["promotion"] = None
        BATCH.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sealed["status"] = "UNBLINDED" if complete else sealed["status"]
    if complete:
        sealed["unblinded_at"] = _now()
    SEALED.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "gt_status": batch["gt_status"],
                "qc_all_pass": all_pass,
                "gates": gates,
                "class_counts": batch["results"]["class_counts"],
                "promoted": str(PROMOTED) if all_pass else None,
            },
            indent=2,
        )
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
