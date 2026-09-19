"""Freeze a fail-closed LORO pair-gate decision from completed fold receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--receipts-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists(): raise FileExistsError(f"Refusing to overwrite immutable LORO decision: {args.output}")
folds = []
for fold in "ABCD":
    receipt = args.receipts_root / f"MV-TRAIN-SEGNEURON-DVID-G2-LORO-{fold}-001" / "receipt.json"
    if not receipt.is_file(): raise ValueError(f"Missing fold receipt: {fold}")
    value = json.loads(receipt.read_text(encoding="utf-8")); best = value["best_held_out_pair_checkpoint"]
    held = best["held_out_region"]
    held_pair_pass = held["same_accuracy_at_0_5"] == 1.0 and held["different_accuracy_at_0_5"] == 1.0 and held["margin"] > 0
    folds.append({"fold": fold, "receipt": str(receipt.resolve()), "receipt_sha256": digest(receipt), "selected_checkpoint": best["checkpoint"], "held_out_region": value["held_out_region"], "held_out_pair_metrics": held, "pair_prerequisite": "PASS" if held_pair_pass else "FAIL", "independent_validation_pair_metrics": best["independent_validation"]})
passed = all(fold["pair_prerequisite"] == "PASS" for fold in folds)
result = {"schema_version": 1, "id": "MV-G2-LORO-PAIR-GATE-001", "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "status": "PASS" if passed else "FAIL", "folds": folds, "decision": {"LORO_RECONSTRUCTION": "NOT_RUN", "reason": "All folds must pass held-out reviewed SAME/DIFFERENT pair prerequisites before raw-inference reconstruction evaluation." if not passed else "Pair prerequisite passed; reconstruction gate required next.", "G2_SPATIAL_GENERALIZATION": "FAIL" if not passed else "PENDING_RECONSTRUCTION", "protected_volume_000004_used": False, "biological_promotions": {"MV-FRAG": 0, "MV-N": 0, "MV-SYN": 0, "MV-CONN": 0}}}
args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "folds": [{"fold": f["fold"], "pair_prerequisite": f["pair_prerequisite"]} for f in folds]}, indent=2))
