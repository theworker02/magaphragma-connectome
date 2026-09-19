"""Finalize DIFFERENT_CLASS_DISCOVERY_001 after blinded review."""
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BATCH = REPO / "experiments/phase6e/DIFFERENT_CLASS_DISCOVERY_001.json"
SEALED = REPO / "experiments/phase6e/DIFFERENT_CLASS_DISCOVERY_AXIS_MAP_SEALED_001.json"
WS_ROOT = REPO / "experiments/phase6e/different-class-discovery-packages"
VALID = {"SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"}
EXCL = {"UNCERTAIN", "BAD_QUESTION"}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def main() -> int:
    batch = json.loads(BATCH.read_text(encoding="utf-8"))
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    if batch.get("verdict"):
        raise SystemExit("Already finalized")
    n_expected = int(batch["design"]["n_decisions"])
    by_opaque = {e["opaque_decision_id"]: e for e in sealed["entries"]}
    decisions = {}
    issues = []
    for crop_dir in WS_ROOT.iterdir():
        if not crop_dir.is_dir():
            continue
        log = crop_dir / "workspace.events.jsonl"
        if not log.exists() or not log.read_text(encoding="utf-8").strip():
            issues.append(f"{crop_dir.name}: empty")
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
                issues.append(f"bad decision {ev.get('decision')}")
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
                "abs_face_step": meta["abs_face_step"],
                "pair_left_zyx": meta["pair_left_zyx"],
                "pair_right_zyx": meta["pair_right_zyx"],
                "source_id": meta["source_id"],
                "event_id": ev["id"],
                "reviewer": ev.get("reviewer"),
            }
    if len(decisions) != n_expected or issues:
        print(json.dumps({"complete": False, "n": len(decisions), "expected": n_expected, "issues": issues}, indent=2))
        return 1

    classable = [d for d in decisions.values() if d["decision"] not in EXCL]
    counts = Counter(d["decision"] for d in classable)
    by_axis = Counter()
    for d in classable:
        by_axis[(d["axis_name"], d["decision"])] += 1
    n_diff = counts["DIFFERENT_PROCESS"]
    if n_diff == 0:
        verdict = "DIFFERENT_CLASS_STILL_NOT_FOUND"
        detail = "Zero DIFFERENT_PROCESS on abs≥12 enriched edges."
    else:
        verdict = "DIFFERENT_CLASS_DEMONSTRATED"
        detail = f"{n_diff} DIFFERENT_PROCESS of {len(classable)} class-countable edges."

    batch["status"] = "COMPLETE_UNBLINDED"
    batch["completed_at"] = _now()
    batch["results"] = {
        "n_decisions": n_expected,
        "class_counts": {
            "SAME_PROCESS": counts["SAME_PROCESS"],
            "DIFFERENT_PROCESS": n_diff,
            "excluded": n_expected - len(classable),
        },
        "by_axis_decision": {f"{a}:{d}": c for (a, d), c in sorted(by_axis.items())},
        "decisions": [decisions[oid] for oid in sorted(decisions)],
    }
    batch["verdict"] = {
        "value": verdict,
        "detail": detail,
        "do_not_train": True,
        "discovery_labels_are_not_automatic_production_gt": True,
    }
    BATCH.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sealed["status"] = "UNBLINDED"
    sealed["unblinded_at"] = _now()
    SEALED.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "detail": detail, "class_counts": batch["results"]["class_counts"], "by_axis": batch["results"]["by_axis_decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
