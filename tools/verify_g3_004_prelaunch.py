"""Pre-launch validation of the -004 stratified Y/TRAIN review batch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS_ROOT = REPO / "experiments/phase6e/g3-external-review-packages-004"
Q_ROOT = REPO / "experiments/phase6e/g3-interface-queues-004"
BAND_LABELS = ["0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"]
PRIOR = {
    "MV-G3-TRAIN-A": ["g3-interface-queues-001/MV-G3-TRAIN-A.json", "g3-interface-queues-003/MV-G3-TRAIN-A.json"],
    "MV-G3-TRAIN-B": ["g3-interface-queues-001/MV-G3-TRAIN-B.json", "g3-interface-queues-003/MV-G3-TRAIN-B.json"],
    "MV-G3-TRAIN-D": ["g3-interface-queues-001/MV-G3-TRAIN-D.json", "g3-interface-queues-003/MV-G3-TRAIN-D.json"],
    "MV-G3-TRAIN-F": ["g3-interface-queues-001/MV-G3-TRAIN-F.json", "g3-interface-queues-003/MV-G3-TRAIN-F.json"],
    "MV-G3-TRAIN-G": ["g3-interface-queues-002/MV-G3-TRAIN-G.json", "g3-interface-queues-003/MV-G3-TRAIN-G.json"],
    "MV-G3-TRAIN-H": ["g3-interface-queues-002/MV-G3-TRAIN-H.json", "g3-interface-queues-003/MV-G3-TRAIN-H.json"],
}
SEP2 = 144


def _pairs(queue: dict) -> set:
    return {(tuple(q["pair_left_zyx"]), tuple(q["pair_right_zyx"]), int(q["channel_zyx"])) for q in queue["questions"]}


def main() -> int:
    failures: list[str] = []
    per_crop = []
    for crop_id, prior_rels in PRIOR.items():
        ws = json.loads((WS_ROOT / crop_id / "workspace.json").read_text())
        q = json.loads((Q_ROOT / f"{crop_id}.json").read_text())

        # fresh empty log + new -004 id
        log = Path(ws["event_log"]["path"])
        if not ws["id"].endswith("-004"):
            failures.append(f"{crop_id}: workspace id not -004")
        if (log.read_text() if log.exists() else "x").strip() != "":
            failures.append(f"{crop_id}: event log not empty")

        # binding + status + raw hash
        if q["workspace_id"] != ws["id"]:
            failures.append(f"{crop_id}: workspace_id mismatch")
        if q["status"] != "EXPERT_INTERFACE_REVIEW_REQUIRED":
            failures.append(f"{crop_id}: bad status")
        raw = Path(ws["raw"]["path"])
        if hashlib.sha256(raw.read_bytes()).hexdigest() != ws["raw"]["sha256"] or q["raw_sha256"] != ws["raw"]["sha256"]:
            failures.append(f"{crop_id}: raw sha mismatch")

        # all Y + TRAIN
        if ws["split"] != "G3_TARGET_TRAIN":
            failures.append(f"{crop_id}: not TRAIN")
        if any(int(qq["channel_zyx"]) != 1 for qq in q["questions"]):
            failures.append(f"{crop_id}: non-Y question present")

        # exclusion vs all prior queues
        prior_all = set()
        for r in prior_rels:
            pth = REPO / "experiments/phase6e" / r
            if pth.exists():
                prior_all |= _pairs(json.loads(pth.read_text()))
        overlap = _pairs(q) & prior_all
        if overlap:
            failures.append(f"{crop_id}: {len(overlap)} pairs overlap prior queues")

        # band/crop allocation matches declared plan (1 interface per band)
        band_counts = {b: 0 for b in BAND_LABELS}
        centres = {}
        for qq in q["questions"]:
            centres.setdefault(qq["interface_id"], qq)
        for iface, qq in centres.items():
            band_counts[qq["band"]] += 1
        if band_counts != {b: 1 for b in BAND_LABELS}:
            failures.append(f"{crop_id}: band allocation {band_counts} != 1 per band")

        # within-batch separation (interface centres = member 1)
        interface_centres = [tuple(qq["pair_left_zyx"]) for qq in q["questions"] if int(qq["interface_member"]) == 2]
        # member 2 is offset (-1,0) from centre; recover centre = member 1
        interface_centres = [tuple(qq["pair_left_zyx"]) for qq in q["questions"] if int(qq["interface_member"]) == 1]
        for i, c in enumerate(interface_centres):
            for d in interface_centres[i + 1:]:
                if (c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2 + (c[2] - d[2]) ** 2 < SEP2:
                    failures.append(f"{crop_id}: within-batch separation violated")
                    break

        # declared rule embedded before labels
        sel = q["selection"]
        if not sel.get("declared_sampling_rule", {}).get("declared_before_labels"):
            failures.append(f"{crop_id}: declared sampling rule metadata missing")

        per_crop.append({"crop_id": crop_id, "interfaces": len(centres), "band_counts": band_counts, "prior_overlap": len(overlap)})

    report = {
        "checks_passed": not failures,
        "failures": failures,
        "total_interfaces": sum(c["interfaces"] for c in per_crop),
        "per_crop": per_crop,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
