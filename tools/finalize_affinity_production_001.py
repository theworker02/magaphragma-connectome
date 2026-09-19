"""Finalize AFFINITY_PRODUCTION_001 (collection complete; still not train-authorized)."""
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BATCH = REPO / "experiments/phase6e/AFFINITY_PRODUCTION_001.json"
SEALED = REPO / "experiments/phase6e/AFFINITY_PRODUCTION_001_AXIS_MAP_SEALED.json"
WS_ROOT = REPO / "experiments/phase6e/affinity-production-001-packages"
VALID = {"SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"}
EXCL = {"UNCERTAIN", "BAD_QUESTION"}
PATH_PATTERN = {"Z": "DIFFERENT_PROCESS", "Y": "SAME_PROCESS", "X": "SAME_PROCESS"}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def main() -> int:
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    if batch.get("completion"):
        raise SystemExit("Already finalized")
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
                "crop_id": crop,
                "source_id": meta["source_id"],
                "center_zyx": meta["center_zyx"],
                "pair_left_zyx": meta["pair_left_zyx"],
                "pair_right_zyx": meta["pair_right_zyx"],
                "channel_zyx": meta["channel_zyx"],
                "face_step": meta.get("face_step"),
                "event_id": ev["id"],
                "reviewer": ev.get("reviewer"),
            }
    if len(decisions) != n_expected or issues:
        print(json.dumps({"complete": False, "n": len(decisions), "issues": issues}, indent=2))
        return 1

    triplets = []
    class_edges = []
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
                "labels_zyx": labels,
                "excluded": excluded,
                "matches_orientation_collinear_pattern": (not excluded) and labels == PATH_PATTERN,
            }
        )

    counts = Counter(e["decision"] for e in class_edges)
    n_same, n_diff = counts["SAME_PROCESS"], counts["DIFFERENT_PROCESS"]
    evaluable = [t for t in triplets if not t["excluded"]]
    collinear = sum(1 for t in evaluable if t["matches_orientation_collinear_pattern"])
    collinear_dominated = bool(evaluable) and collinear == len(evaluable)

    batch["status"] = "COLLECTION_COMPLETE_UNBLINDED"
    batch["completed_at"] = _now()
    batch["results"] = {
        "n_decisions": n_expected,
        "class_counts": {
            "SAME_PROCESS": n_same,
            "DIFFERENT_PROCESS": n_diff,
            "excluded": n_expected - n_same - n_diff,
        },
        "triplets": triplets,
        "collinear_evaluable_fraction": (collinear / len(evaluable)) if evaluable else None,
        "decisions": [decisions[oid] for oid in sorted(decisions)],
    }
    batch["completion"] = {
        "both_classes_present": n_same >= 1 and n_diff >= 1,
        "orientation_collinear_dominated": collinear_dominated,
        "eligible_for_train_authorization_review": (
            n_same >= 1 and n_diff >= 1 and not collinear_dominated
        ),
        "do_not_train": True,
        "requires_separate_TRAIN_AUTHORIZATION_artifact": True,
        "note": (
            "Collection complete under validated procedure. "
            "Do not fit models until TRAIN_AUTHORIZATION_* is written."
        ),
    }
    BATCH.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sealed["status"] = "UNBLINDED"
    sealed["unblinded_at"] = _now()
    SEALED.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": batch["status"],
                "class_counts": batch["results"]["class_counts"],
                "completion": batch["completion"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
