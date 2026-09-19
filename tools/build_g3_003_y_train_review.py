"""Build fresh -003 review workspaces + Y-oriented TRAIN interface queues.

Purpose: the frozen axis-balanced cohort gate (MV-G3-AXIS-REVIEW-COHORT-002)
failed closed because axis Y is missing its TRAIN x DIFFERENT_PROCESS cell.
This creates genuinely new, previously unreviewed, LABEL-BLIND Y-oriented
candidates on the TRAIN crops so an expert can review them.

Immutability/provenance rules honored here:
  * Never reuse an existing workspace or its event log. Each -003 workspace has
    a new ID and a fresh, empty append-only event log.
  * Reference the same immutable raw crop (path + sha256) as the source -001/-002
    workspace; the raw voxels are never copied or altered.
  * The -003 interface queue is bound exclusively to its -003 workspace
    (workspace_id + raw_sha256), and excludes every exact pair already asked in
    the prior -001/-002 queue.
  * Refuse to overwrite any existing artifact.

Candidate selection is delegated to generate_g3_interface_queue.py with
--orientation-axis Y and --exclude-queue <prior>. It reads raw EM geometry only
and never consults SAME/DIFFERENT decisions.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "tools" / "generate_g3_interface_queue.py"

# The six frozen G3 TRAIN crops and their prior interface queue (the exact
# pairs of which must be excluded from fresh review).
TRAIN_SOURCES = {
    "MV-G3-TRAIN-A": ("g3-external-review-packages-001", "g3-interface-queues-001/MV-G3-TRAIN-A.json"),
    "MV-G3-TRAIN-B": ("g3-external-review-packages-001", "g3-interface-queues-001/MV-G3-TRAIN-B.json"),
    "MV-G3-TRAIN-D": ("g3-external-review-packages-001", "g3-interface-queues-001/MV-G3-TRAIN-D.json"),
    "MV-G3-TRAIN-F": ("g3-external-review-packages-001", "g3-interface-queues-001/MV-G3-TRAIN-F.json"),
    "MV-G3-TRAIN-G": ("g3-external-review-packages-002", "g3-interface-queues-002/MV-G3-TRAIN-G.json"),
    "MV-G3-TRAIN-H": ("g3-external-review-packages-002", "g3-interface-queues-002/MV-G3-TRAIN-H.json"),
}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_workspace(crop_id: str, source_pkg: str, out_root: Path) -> Path:
    source_ws_path = REPO / "experiments/phase6e" / source_pkg / crop_id / "workspace.json"
    source = json.loads(source_ws_path.read_text(encoding="utf-8"))
    if source.get("split") != "G3_TARGET_TRAIN":
        raise ValueError(f"{crop_id}: source workspace is not TRAIN")
    crop_dir = out_root / crop_id
    if crop_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing -003 workspace dir: {crop_dir}")
    crop_dir.mkdir(parents=True)
    event_log = crop_dir / "workspace.events.jsonl"
    # Fresh, empty append-only log. Genuinely new workspace identity.
    event_log.write_text("", encoding="utf-8")
    workspace = {
        "allowed_decisions": ["BAD_QUESTION", "DIFFERENT_PROCESS", "SAME_PROCESS", "UNCERTAIN"],
        "coordinate_frame": source.get("coordinate_frame"),
        "created_at": _now(),
        "crop_id": crop_id,
        "event_log": {"append_only": True, "path": str(event_log.resolve())},
        "id": f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id[-1]}-003",
        "pair_contract": source.get("pair_contract"),
        "parent_region_id": source.get("parent_region_id"),
        "prohibited_promotions": source.get("prohibited_promotions"),
        # Same immutable raw crop, byte-identical path + hash.
        "raw": source["raw"],
        "provenance": {
            "derived_from_workspace": {"path": str(source_ws_path.resolve()), "id": source["id"]},
            "reason": "Fresh Y-oriented TRAIN review to fill axis Y x TRAIN x DIFFERENT_PROCESS deficit",
        },
        "review_state": "UNREVIEWED",
        "schema_version": 1,
        "source_origin_xyz": source.get("source_origin_xyz"),
        "split": "G3_TARGET_TRAIN",
        "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
    }
    workspace_path = crop_dir / "workspace.json"
    workspace_path.write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return workspace_path


def build_queue(crop_id: str, workspace_path: Path, prior_queue_rel: str, interfaces: int, out_root: Path) -> dict:
    prior_queue = REPO / "experiments/phase6e" / prior_queue_rel
    queue_out = out_root / f"{crop_id}.json"
    cmd = [
        sys.executable, str(GENERATOR), str(workspace_path),
        "--output", str(queue_out),
        "--interfaces", str(interfaces),
        "--orientation-axis", "Y",
        "--exclude-queue", str(prior_queue),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    return {"crop_id": crop_id, "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "queue": str(queue_out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interfaces-per-crop", type=int, default=4)
    parser.add_argument("--workspace-root", type=Path, default=REPO / "experiments/phase6e/g3-external-review-packages-003")
    parser.add_argument("--queue-root", type=Path, default=REPO / "experiments/phase6e/g3-interface-queues-003")
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to specific crop ids")
    args = parser.parse_args()
    crops = args.only or list(TRAIN_SOURCES)
    results = []
    for crop_id in crops:
        source_pkg, prior_queue_rel = TRAIN_SOURCES[crop_id]
        workspace_path = build_workspace(crop_id, source_pkg, args.workspace_root)
        outcome = build_queue(crop_id, workspace_path, prior_queue_rel, args.interfaces_per_crop, args.queue_root)
        outcome["workspace"] = str(workspace_path.resolve())
        results.append(outcome)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
