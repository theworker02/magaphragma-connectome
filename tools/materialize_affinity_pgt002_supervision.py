"""Materialize frozen PGT002 train/eval sparse affinity manifests (no training).

Builds per-split edge lists from PRODUCTION_GT_PROMOTED_BATCH_002 under the
frozen AFFINITY_TRAINING_EXPERIMENT_CONTRACT_001 split. Does not require both
classes inside each individual crop receipt.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROMOTED = REPO / "experiments/phase6e/PRODUCTION_GT_PROMOTED_BATCH_002.json"
CONTRACT = REPO / "experiments/phase6e/AFFINITY_TRAINING_EXPERIMENT_CONTRACT_001.json"
OUT = REPO / "experiments/phase6e/AFFINITY-TRAIN-PGT002-001/supervision"
MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"Refusing overwrite: {OUT}")
    promoted = json.loads(PROMOTED.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    file_sha = sha256(PROMOTED)
    if file_sha != contract["eligible_gt"]["promoted_artifact_sha256"]:
        raise SystemExit(f"Promoted GT hash mismatch: {file_sha}")
    if promoted.get("status") != "PRODUCTION_GT_PROMOTED":
        raise SystemExit("GT not promoted")

    records = {r["id"]: r for r in json.loads(MANIFEST.read_text(encoding="utf-8"))["records"]}
    train_sources = set(contract["train_eval_split"]["train_sources"])
    eval_sources = set(contract["train_eval_split"]["eval_sources"])
    if train_sources & eval_sources:
        raise SystemExit("train/eval leakage in contract")

    allow = set(contract["eligible_gt"]["label_filter"])
    drop = set(contract["eligible_gt"]["drop"])

    def edges_for(sources: set[str]) -> list[dict]:
        out = []
        for lab in promoted["labels"]:
            if lab["decision"] in drop:
                continue
            if lab["decision"] not in allow:
                continue
            sid = lab["source_id"]
            if sid not in sources:
                continue
            rec = records[sid]
            if rec["raw_sha256"] != lab["raw_sha256"]:
                raise SystemExit(f"raw hash mismatch for {sid}")
            out.append(
                {
                    "opaque_decision_id": lab["opaque_decision_id"],
                    "source_id": sid,
                    "crop_id": lab["crop_id"],
                    "raw_path": rec["raw_path"],
                    "raw_sha256": lab["raw_sha256"],
                    "shape_zyx": rec["shape_zyx"],
                    "channel_zyx": lab["channel_zyx"],
                    "pair_left_zyx": lab["pair_left_zyx"],
                    "pair_right_zyx": lab["pair_right_zyx"],
                    "decision": lab["decision"],
                    "target": 1 if lab["decision"] == "SAME_PROCESS" else 0,
                    "axis_name": lab["axis_name"],
                }
            )
        return out

    train_edges = edges_for(train_sources)
    eval_edges = edges_for(eval_sources)
    if not train_edges or not eval_edges:
        raise SystemExit("empty train or eval")
    if {e["source_id"] for e in train_edges} != train_sources:
        raise SystemExit("train sources incomplete")
    if {e["source_id"] for e in eval_edges} != eval_sources:
        raise SystemExit("eval sources incomplete")

    OUT.mkdir(parents=True)
    payload = {
        "id": "AFFINITY-TRAIN-PGT002-001-SUPERVISION",
        "created_at": _now(),
        "status": "FROZEN_SPARSE_EDGE_SUPERVISION",
        "contract_id": "AFFINITY_TRAINING_EXPERIMENT_CONTRACT_001",
        "promoted_artifact": str(PROMOTED.relative_to(REPO)).replace("\\", "/"),
        "promoted_artifact_sha256": file_sha,
        "train_sources": sorted(train_sources),
        "eval_sources": sorted(eval_sources),
        "train_edges": train_edges,
        "eval_edges": eval_edges,
        "counts": {
            "train": {
                "n": len(train_edges),
                "SAME_PROCESS": sum(e["decision"] == "SAME_PROCESS" for e in train_edges),
                "DIFFERENT_PROCESS": sum(e["decision"] == "DIFFERENT_PROCESS" for e in train_edges),
            },
            "eval": {
                "n": len(eval_edges),
                "SAME_PROCESS": sum(e["decision"] == "SAME_PROCESS" for e in eval_edges),
                "DIFFERENT_PROCESS": sum(e["decision"] == "DIFFERENT_PROCESS" for e in eval_edges),
            },
        },
    }
    path = OUT / "supervision_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "counts": payload["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
