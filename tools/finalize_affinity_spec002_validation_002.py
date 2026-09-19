"""Finalize AFFINITY_SPEC002_VALIDATION_002 with PROTOCOL_001 verdict rules."""
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BATCH = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_002.json"
SEALED = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_002_AXIS_MAP_SEALED.json"
PROTOCOL = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_PROTOCOL_001.json"
WS_ROOT = REPO / "experiments/phase6e/affinity-spec002-validation-002-packages"
VALID = {"SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"}
EXCL = {"UNCERTAIN", "BAD_QUESTION"}
N_EXPECTED = 36
PATH_PATTERN = {"Z": "DIFFERENT_PROCESS", "Y": "SAME_PROCESS", "X": "SAME_PROCESS"}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def orientation_invariance_failed(triplets: list[dict], class_edges: list[dict]) -> tuple[bool, str]:
    evaluable = [t for t in triplets if not t["excluded_from_class_counts"]]
    if evaluable and all(t["matches_orientation_collinear_pattern"] for t in evaluable):
        return True, f"All {len(evaluable)} evaluable triplets match Z=DIFF & Y=SAME & X=SAME."
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
            if all(x == "DIFFERENT_PROCESS" for x in labels) and "SAME_PROCESS" not in labels:
                if all(x == "SAME_PROCESS" for x in ol) and "DIFFERENT_PROCESS" not in ol:
                    return True, f"Axis {ax} all DIFFERENT while {oa} all SAME."
            if all(x == "SAME_PROCESS" for x in labels) and "DIFFERENT_PROCESS" not in labels:
                if all(x == "DIFFERENT_PROCESS" for x in ol) and "SAME_PROCESS" not in ol:
                    return True, f"Axis {ax} all SAME while {oa} all DIFFERENT."
    return False, ""


def apply_verdict(n_same: int, n_diff: int, oif: bool, oif_detail: str) -> tuple[str, str]:
    if n_diff == 0:
        return "NEGATIVE_CLASS_NOT_DEMONSTRATED", "Zero DIFFERENT_PROCESS among class-countable edges."
    if oif:
        return "ORIENTATION_INVARIANCE_FAILED", oif_detail
    if n_same >= 1 and n_diff >= 1:
        return (
            "SPEC002_ANNOTATION_PROCEDURE_VALIDATED",
            f"Both classes present (SAME={n_same}, DIFFERENT={n_diff}); orientation invariance not failed.",
        )
    return "SPEC002_PROCEDURE_UNRESOLVED", f"Ambiguous (SAME={n_same}, DIFFERENT={n_diff})."


def main() -> int:
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if batch.get("verdict"):
        raise SystemExit("Already finalized")
    by_opaque = {e["opaque_decision_id"]: e for e in sealed["entries"]}
    decisions: dict[str, dict] = {}
    issues: list[str] = []
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
            if oid not in by_opaque:
                issues.append(f"unknown {oid}")
                continue
            if ev.get("decision") not in VALID:
                issues.append(f"bad {ev.get('decision')}")
                continue
            meta = by_opaque[oid]
            if list(ev["pair_left_zyx"]) != meta["pair_left_zyx"] or int(ev["channel_zyx"]) != meta["channel_zyx"]:
                issues.append(f"{oid}: geometry mismatch")
            if oid in decisions:
                issues.append(f"{oid}: duplicate")
            decisions[oid] = {
                "opaque_decision_id": oid,
                "decision": ev["decision"],
                "event_id": ev["id"],
                "crop_id": crop,
                "source_id": meta["source_id"],
                "center_zyx": meta["center_zyx"],
                "axis_name": meta["axis_name"],
                "channel_zyx": meta["channel_zyx"],
                "pair_left_zyx": meta["pair_left_zyx"],
                "pair_right_zyx": meta["pair_right_zyx"],
                "face_step": meta.get("face_step"),
                "reviewer": ev.get("reviewer"),
            }
    if len(decisions) != N_EXPECTED or issues:
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
                class_edges.append({"opaque_decision_id": oid, "axis_name": meta["axis_name"], "decision": d, "crop_id": crop})
        triplets.append(
            {
                "crop_id": crop,
                "source_id": loc["source_id"],
                "center_zyx": loc["center_zyx"],
                "labels_zyx": labels,
                "excluded_from_class_counts": excluded,
                "matches_orientation_collinear_pattern": (not excluded) and labels == PATH_PATTERN,
            }
        )

    counts = Counter(e["decision"] for e in class_edges)
    n_same, n_diff = counts["SAME_PROCESS"], counts["DIFFERENT_PROCESS"]
    oif, oif_detail = orientation_invariance_failed(triplets, class_edges)
    verdict, detail = apply_verdict(n_same, n_diff, oif, oif_detail)

    batch["status"] = "COMPLETE_UNBLINDED"
    batch["completed_at"] = _now()
    batch["results"] = {
        "n_decisions": N_EXPECTED,
        "class_counts": {
            "SAME_PROCESS": n_same,
            "DIFFERENT_PROCESS": n_diff,
            "excluded_UNCERTAIN_or_BAD": N_EXPECTED - n_same - n_diff,
        },
        "triplets": triplets,
        "decisions": [decisions[oid] for oid in sorted(decisions)],
        "orientation_invariance_failed": oif,
        "orientation_invariance_detail": oif_detail or None,
    }
    batch["verdict"] = {
        "value": verdict,
        "detail": detail,
        "rules_source": "AFFINITY_SPEC002_VALIDATION_PROTOCOL_001.predeclared_verdict_rules",
        "rules_match_protocol": batch["predeclared_verdict_rules"] == protocol["predeclared_verdict_rules"],
        "do_not_train": True,
        "not_automatic_production_gt": True,
    }
    BATCH.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sealed["status"] = "UNBLINDED"
    sealed["unblinded_at"] = _now()
    SEALED.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "detail": detail, "class_counts": batch["results"]["class_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
