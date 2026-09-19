"""Pre-launch validation for the -005 new-TRAIN review batch (crops I-L)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS_ROOT = REPO / "experiments/phase6e/g3-external-review-packages-005"
Q_ROOT = REPO / "experiments/phase6e/g3-interface-queues-005"
PROTOCOL = json.loads((REPO / "experiments/phase6e/G3_NEW_REGION_SELECTION_PROTOCOL_001.json").read_text())
SURVEY = json.loads((REPO / "experiments/phase6e/MV-G3-NEW-REGION-SURVEY-001.json").read_text())
SEP2 = 144
BANDS = ["0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"]


def _overlap(a, b):
    return all(max(a[x][0], b[x][0]) < min(a[x][1], b[x][1]) for x in ("x", "y", "z"))


def main() -> int:
    failures = []
    per_crop = []
    selected = {s["new_crop_id"]: s for s in PROTOCOL["selected"]}
    existing_boxes = SURVEY["existing_region_boxes"]

    for crop_id, sel in selected.items():
        ws = json.loads((WS_ROOT / crop_id / "workspace.json").read_text())
        q = json.loads((Q_ROOT / f"{crop_id}.json").read_text())

        # spatial independence vs every existing region
        for other in existing_boxes:
            if _overlap(sel["bounds_xyz"], other["bounds_xyz"]):
                failures.append(f"{crop_id}: overlaps existing {other['crop_id']}")
        # not an excluded/used source
        if sel["source_id"] in SURVEY["excluded_existing_sources"]:
            failures.append(f"{crop_id}: source is an already-used region")
        # TRAIN split
        if ws["split"] != "G3_TARGET_TRAIN":
            failures.append(f"{crop_id}: not TRAIN")
        # fresh empty log + new id
        log = Path(ws["event_log"]["path"])
        if not ws["id"].endswith("-005"):
            failures.append(f"{crop_id}: workspace id not -005")
        if (log.read_text() if log.exists() else "x").strip() != "":
            failures.append(f"{crop_id}: event log not empty")
        # queue binding + raw hash integrity
        if q["workspace_id"] != ws["id"]:
            failures.append(f"{crop_id}: workspace_id mismatch")
        if q["status"] != "EXPERT_INTERFACE_REVIEW_REQUIRED":
            failures.append(f"{crop_id}: bad status")
        raw = Path(ws["raw"]["path"])
        actual = hashlib.sha256(raw.read_bytes()).hexdigest()
        if actual != ws["raw"]["sha256"] or q["raw_sha256"] != ws["raw"]["sha256"] or actual != sel["raw_sha256"]:
            failures.append(f"{crop_id}: raw sha mismatch (disk/ws/queue/protocol)")
        # label-blind sampler recorded
        rule = q["selection"].get("declared_sampling_rule", {})
        if not rule.get("declared_before_labels") or "SAME/DIFFERENT labels" not in rule.get("forbidden_selection_inputs", []):
            failures.append(f"{crop_id}: sampler not recorded as label-blind")
        # axis coverage present
        centres = {}
        for qq in q["questions"]:
            centres.setdefault(qq["interface_id"], qq)
        axis_counts = {"Z": 0, "Y": 0, "X": 0}
        for iface, qq in centres.items():
            axis_counts[qq["axis_name"]] += 1
        if not all(axis_counts[a] > 0 for a in ("Z", "Y", "X")):
            failures.append(f"{crop_id}: missing an axis in coverage: {axis_counts}")
        # within-crop separation of interface centres (member 1)
        ic = [tuple(qq["pair_left_zyx"]) for qq in q["questions"] if int(qq["interface_member"]) == 1]
        for i, c in enumerate(ic):
            for d in ic[i+1:]:
                if (c[0]-d[0])**2 + (c[1]-d[1])**2 + (c[2]-d[2])**2 < SEP2:
                    failures.append(f"{crop_id}: within-crop separation violated")
                    break
        per_crop.append({"crop_id": crop_id, "source_id": sel["source_id"], "interfaces": len(centres), "axis_counts": axis_counts, "min_gap_to_existing": sel["min_gap_to_any_existing"]})

    report = {"checks_passed": not failures, "failures": failures, "crops": len(per_crop), "per_crop": per_crop}
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
