"""Pre-launch validation of the -003 Y-oriented TRAIN review artifacts.

Checks the nine conditions required before launching Napari:
 1. each -003 workspace has a new ID and a fresh/empty event log;
 2. the -003 queue workspace_id exactly matches its workspace;
 3. queue raw_sha256 matches the immutable raw crop;
 4. queue status is EXPERT_INTERFACE_REVIEW_REQUIRED;
 5. every queued interface is Y-oriented (channel_zyx == 1) and TRAIN;
 6. no exact pair overlaps the excluded -001/-002 queue;
 7. selection did not consume review labels (recorded label-blind contract);
 8. (run separately) regression tests pass;
 9. artifacts refuse overwrite (existence => refusal at generation time).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS_ROOT = REPO / "experiments/phase6e/g3-external-review-packages-003"
Q_ROOT = REPO / "experiments/phase6e/g3-interface-queues-003"
PRIOR = {
    "MV-G3-TRAIN-A": REPO / "experiments/phase6e/g3-interface-queues-001/MV-G3-TRAIN-A.json",
    "MV-G3-TRAIN-B": REPO / "experiments/phase6e/g3-interface-queues-001/MV-G3-TRAIN-B.json",
    "MV-G3-TRAIN-D": REPO / "experiments/phase6e/g3-interface-queues-001/MV-G3-TRAIN-D.json",
    "MV-G3-TRAIN-F": REPO / "experiments/phase6e/g3-interface-queues-001/MV-G3-TRAIN-F.json",
    "MV-G3-TRAIN-G": REPO / "experiments/phase6e/g3-interface-queues-002/MV-G3-TRAIN-G.json",
    "MV-G3-TRAIN-H": REPO / "experiments/phase6e/g3-interface-queues-002/MV-G3-TRAIN-H.json",
}
PRIOR_WS_IDS = {
    "RAIN-A", "RAIN-B", "RAIN-D", "RAIN-F",
    "MV-EXTERNAL-BOUNDARY-WORKSPACE-RAIN-A", "MV-EXTERNAL-BOUNDARY-WORKSPACE-RAIN-B",
}


def _pairs(queue: dict) -> set:
    return {(tuple(q["pair_left_zyx"]), tuple(q["pair_right_zyx"]), int(q["channel_zyx"])) for q in queue["questions"]}


def main() -> int:
    failures: list[str] = []
    summary = []
    total_interfaces = 0
    total_questions = 0
    for crop_id, prior_path in PRIOR.items():
        ws_path = WS_ROOT / crop_id / "workspace.json"
        q_path = Q_ROOT / f"{crop_id}.json"
        ws = json.loads(ws_path.read_text())
        q = json.loads(q_path.read_text())
        prior = json.loads(prior_path.read_text())

        # (1) new ID + fresh/empty event log
        log_path = Path(ws["event_log"]["path"])
        log_text = log_path.read_text() if log_path.exists() else "<missing>"
        if ws["id"] in PRIOR_WS_IDS or not ws["id"].endswith("-003"):
            failures.append(f"{crop_id}: workspace id not new: {ws['id']}")
        if log_text.strip() != "":
            failures.append(f"{crop_id}: event log is not empty")

        # (2) queue workspace_id matches workspace
        if q["workspace_id"] != ws["id"]:
            failures.append(f"{crop_id}: queue workspace_id {q['workspace_id']} != workspace {ws['id']}")

        # (3) raw_sha256 matches immutable raw crop
        raw_path = Path(ws["raw"]["path"])
        actual_raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if q["raw_sha256"] != ws["raw"]["sha256"] or actual_raw_sha != ws["raw"]["sha256"]:
            failures.append(f"{crop_id}: raw sha mismatch (queue/ws/disk)")

        # (4) status
        if q["status"] != "EXPERT_INTERFACE_REVIEW_REQUIRED":
            failures.append(f"{crop_id}: queue status {q['status']}")

        # (5) all Y-oriented + TRAIN
        if ws["split"] != "G3_TARGET_TRAIN":
            failures.append(f"{crop_id}: workspace split {ws['split']}")
        non_y = [qq["id"] for qq in q["questions"] if int(qq["channel_zyx"]) != 1]
        if non_y:
            failures.append(f"{crop_id}: non-Y questions present: {non_y[:3]}")

        # (6) no overlap with excluded prior pairs
        overlap = _pairs(q) & _pairs(prior)
        if overlap:
            failures.append(f"{crop_id}: {len(overlap)} pairs overlap prior queue")

        # (7) label-blind contract recorded
        sel = q.get("selection", {})
        if not sel.get("label_blind") or "reviewed SAME/DIFFERENT decisions" not in sel.get("prohibited_inputs", []):
            failures.append(f"{crop_id}: selection does not record label-blind contract")

        interfaces = len({qq["interface_id"] for qq in q["questions"]})
        total_interfaces += interfaces
        total_questions += len(q["questions"])
        summary.append({
            "crop_id": crop_id,
            "workspace_id": ws["id"],
            "interfaces": interfaces,
            "questions": len(q["questions"]),
            "all_Y": not non_y,
            "prior_overlap": len(overlap),
            "excluded_prior_pairs": sel.get("excluded_pair_count"),
        })

    report = {
        "checks_passed": not failures,
        "failures": failures,
        "total_Y_train_interfaces": total_interfaces,
        "total_Y_train_questions": total_questions,
        "per_crop": summary,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
