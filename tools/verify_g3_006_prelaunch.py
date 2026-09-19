"""Pre-launch validation for the -006 axis-neutral paired-location batch."""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS_ROOT = REPO / "experiments/phase6e/g3-external-review-packages-006"
Q_ROOT = REPO / "experiments/phase6e/g3-interface-queues-006"
SURVEY = json.loads((REPO / "experiments/phase6e/MV-G3-NEW-REGION-SURVEY-001.json").read_text())
SEP2 = 144


def _overlap(a, b):
    return all(max(a[x][0], b[x][0]) < min(a[x][1], b[x][1]) for x in ("x", "y", "z"))


def _all_prior_edges():
    edges = set()
    for qp in glob.glob(str(REPO / "experiments/phase6e/g3-interface-queues-*/MV-G3-*.json")):
        if "queues-006" in qp:
            continue
        try:
            q = json.loads(Path(qp).read_text())
        except Exception:
            continue
        for question in q.get("questions", []):
            edges.add((tuple(question["pair_left_zyx"]), tuple(question["pair_right_zyx"]), int(question["channel_zyx"])))
    return edges


def main() -> int:
    failures = []
    prior = _all_prior_edges()
    existing_boxes = SURVEY["existing_region_boxes"]
    src_by_id = {e["candidate_source_id"]: e for e in SURVEY["surveyed_population"]}
    per_crop = []
    for wf in sorted(glob.glob(str(WS_ROOT / "*/workspace.json"))):
        ws = json.loads(Path(wf).read_text())
        crop_id = ws["crop_id"]
        q = json.loads((Q_ROOT / f"{crop_id}.json").read_text())
        src = src_by_id[ws["parent_region_id"]]
        # spatial independence
        for other in existing_boxes:
            if _overlap(src["bounds_xyz"], other["bounds_xyz"]):
                failures.append(f"{crop_id}: overlaps existing {other['crop_id']}")
        # fresh empty log + -006 id + TRAIN
        log = Path(ws["event_log"]["path"])
        if not ws["id"].endswith("-006"):
            failures.append(f"{crop_id}: id not -006")
        if (log.read_text() if log.exists() else "x").strip() != "":
            failures.append(f"{crop_id}: log not empty")
        if ws["split"] != "G3_TARGET_TRAIN":
            failures.append(f"{crop_id}: not TRAIN")
        # binding + raw hash
        if q["workspace_id"] != ws["id"]:
            failures.append(f"{crop_id}: workspace_id mismatch")
        raw = Path(ws["raw"]["path"])
        if hashlib.sha256(raw.read_bytes()).hexdigest() != ws["raw"]["sha256"] or q["raw_sha256"] != ws["raw"]["sha256"]:
            failures.append(f"{crop_id}: raw sha mismatch")
        # equal axis representation + all 3 edges per location
        by_loc = {}
        for qq in q["questions"]:
            by_loc.setdefault(qq["location_index"], []).append(int(qq["channel_zyx"]))
        for li, axes in by_loc.items():
            if sorted(axes) != [0, 1, 2]:
                failures.append(f"{crop_id}: location {li} axes {sorted(axes)} != [0,1,2]")
        axis_counts = {0: 0, 1: 0, 2: 0}
        for qq in q["questions"]:
            axis_counts[int(qq["channel_zyx"])] += 1
        if not (axis_counts[0] == axis_counts[1] == axis_counts[2]):
            failures.append(f"{crop_id}: unequal axis representation {axis_counts}")
        # exclusion vs all prior edges
        my_edges = {(tuple(qq["pair_left_zyx"]), tuple(qq["pair_right_zyx"]), int(qq["channel_zyx"])) for qq in q["questions"]}
        overlap = my_edges & prior
        if overlap:
            failures.append(f"{crop_id}: {len(overlap)} edges overlap prior queues")
        # location-center separation
        centres = [tuple(qq["pair_left_zyx"]) for qq in q["questions"] if int(qq["channel_zyx"]) == 0]
        for i, c in enumerate(centres):
            for d in centres[i+1:]:
                if (c[0]-d[0])**2 + (c[1]-d[1])**2 + (c[2]-d[2])**2 < SEP2:
                    failures.append(f"{crop_id}: center separation violated")
                    break
        # interleave: not all-Z first
        first3 = [q["questions"][i]["axis_name"] for i in range(min(3, len(q["questions"])))]
        if len(set(first3)) == 1:
            failures.append(f"{crop_id}: presentation not interleaved (first 3 all {first3[0]})")
        per_crop.append({"crop_id": crop_id, "source_id": ws["parent_region_id"], "edges": len(q["questions"]), "axis_counts": axis_counts, "prior_overlap": len(overlap)})
    report = {"checks_passed": not failures, "failures": failures, "per_crop": per_crop}
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
