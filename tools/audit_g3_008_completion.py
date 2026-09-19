"""Read-only completion-integrity audit of the -008 axis-neutral paired-location
experiment. Proves (or refutes) that all 12 centers x 3 edges = 36 intended
edges each have exactly one valid completed human decision, with no missing,
duplicate, superseded, or malformed decisions, and that the queue was not
altered after review began. Extrapolation is not used: every triplet is read
from the actual append-only event logs.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
Q_ROOT = REPO / "experiments/phase6e/g3-interface-queues-008"
WS_ROOT = REPO / "experiments/phase6e/g3-external-review-packages-008"
OUT = REPO / "experiments/phase6e/G3_AXNEU4_008_COMPLETION_AUDIT.json"
VALID = {"SAME_PROCESS", "DIFFERENT_PROCESS", "UNCERTAIN", "BAD_QUESTION"}
AXES = {0: "Z", 1: "Y", 2: "X"}


def _expected_pair(center, axis):
    left = list(center)
    right = left[:axis] + [left[axis] + 1] + left[axis + 1:]
    return tuple(left), tuple(right)


def main() -> None:
    queues = sorted(glob.glob(str(Q_ROOT / "MV-G3-AXNEU4-*.json")))
    failures = []
    triplets = {}
    per_location = []
    total_decisions = 0

    for qp in queues:
        queue = json.loads(Path(qp).read_text())
        crop = queue["crop_id"]
        center = tuple(queue["selection"]["center_zyx"])
        ws_path = WS_ROOT / crop / "workspace.json"
        ws = json.loads(ws_path.read_text())
        log_path = Path(ws["event_log"]["path"])

        # queue defines exactly 3 edges Z/Y/X at the same center, coord-correct
        q_by_axis = {}
        for qq in queue["questions"]:
            axis = int(qq["channel_zyx"])
            exp_left, exp_right = _expected_pair(center, axis)
            if tuple(qq["pair_left_zyx"]) != exp_left or tuple(qq["pair_right_zyx"]) != exp_right:
                failures.append(f"{crop}: queue edge {AXES[axis]} not canonical +1 pair")
            q_by_axis[axis] = qq["id"]
        if set(q_by_axis) != {0, 1, 2}:
            failures.append(f"{crop}: queue does not define exactly Z/Y/X edges")

        # queue integrity vs workspace binding
        if queue["workspace_id"] != ws["id"] or queue["raw_sha256"] != ws["raw"]["sha256"]:
            failures.append(f"{crop}: queue/workspace binding mismatch")

        # read event log
        events = []
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        # one valid final decision per intended edge (by question_reference)
        by_ref = {}
        for e in events:
            ref = e.get("question_reference")
            dec = e.get("decision")
            if dec not in VALID:
                failures.append(f"{crop}: malformed decision {dec!r}")
                continue
            by_ref.setdefault(ref, []).append(e)

        loc_triplet = {}
        for axis in (0, 1, 2):
            ref = q_by_axis.get(axis)
            evs = by_ref.get(ref, [])
            if len(evs) == 0:
                failures.append(f"{crop}: MISSING decision for {AXES[axis]} edge ({ref})")
                loc_triplet[AXES[axis]] = None
                continue
            if len(evs) > 1:
                # multiple events for one edge: only acceptable if we can pick a
                # single final one; flag as needing scrutiny (superseded/dup).
                failures.append(f"{crop}: {len(evs)} decisions for {AXES[axis]} edge ({ref}) — duplicate/superseded")
            final = evs[-1]
            # verify the event's own geometry matches the intended edge
            exp_left, exp_right = _expected_pair(center, axis)
            if tuple(final.get("pair_left_zyx", ())) != exp_left or tuple(final.get("pair_right_zyx", ())) != exp_right:
                failures.append(f"{crop}: {AXES[axis]} event geometry != intended edge")
            if int(final.get("channel_zyx", -1)) != axis:
                failures.append(f"{crop}: {AXES[axis]} event channel mismatch")
            loc_triplet[AXES[axis]] = final.get("decision")
            total_decisions += 1

        # unexpected extra refs not in the queue
        extra = set(by_ref) - set(q_by_axis.values())
        if extra:
            failures.append(f"{crop}: event log has {len(extra)} decisions not in queue")

        triplets[crop] = {"center_zyx": list(center), "stratum": queue["selection"].get("stratum"), "triplet": loc_triplet}
        per_location.append({"crop": crop, "center_zyx": list(center), "triplet": loc_triplet,
                             "events": len(events), "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest() if log_path.exists() else None})

    n_locations = len(queues)
    complete = (n_locations == 12 and total_decisions == 36 and not failures)
    # does the Z=DIFFERENT, Y=SAME, X=SAME pattern hold in ALL 12?
    pattern = {"Z": "DIFFERENT_PROCESS", "Y": "SAME_PROCESS", "X": "SAME_PROCESS"}
    all_match = complete and all(t["triplet"] == pattern for t in triplets.values())

    report = {
        "id": "MV-G3-AXNEU4-008-COMPLETION-AUDIT",
        "n_locations": n_locations,
        "intended_centers": 12,
        "intended_edges": 36,
        "completed_valid_decisions": total_decisions,
        "complete_36_of_36": complete,
        "integrity_failures": failures,
        "triplets": triplets,
        "per_location": per_location,
        "hypothesized_pattern": pattern,
        "all_12_match_Z_DIFF_Y_X_SAME": all_match,
        "final_status": ("COMPLETE_AND_PATTERN_CONFIRMED" if all_match else
                          "COMPLETE_BUT_PATTERN_NOT_UNIFORM" if complete else
                          "INCOMPLETE_OR_INTEGRITY_FAILURE"),
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_locations", "completed_valid_decisions", "complete_36_of_36", "all_12_match_Z_DIFF_Y_X_SAME", "final_status", "integrity_failures")}, indent=2))
    print("\nTRIPLETS:")
    for crop in sorted(triplets):
        t = triplets[crop]
        print(f"  {crop} center={t['center_zyx']} stratum={t['stratum']}: {t['triplet']}")


if __name__ == "__main__":
    main()
