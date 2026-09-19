"""Regression coverage for frozen reviewed G2 affinity supervision."""

import json
from pathlib import Path

import numpy as np
import pytest

from mvconnectome.g2_supervision import freeze_g2_supervision
from mvconnectome.io import sha256_file


def _receipt(root: Path, region: dict, *, different_at_edge: bool = False) -> None:
    out = root / region["id"]; out.mkdir(parents=True)
    raw = out / "raw.npy"; np.save(raw, np.zeros((4, 5, 6), dtype=np.uint8))
    targets = out / "targets.npy"; mask = out / "mask.npy"
    np.save(targets, np.zeros((3, 4, 5, 6), dtype=np.uint8)); np.save(mask, np.zeros((3, 4, 5, 6), dtype=np.uint8))
    log = out / "events.jsonl"
    events = [
        {"decision": "SAME_PROCESS", "reviewer": "reviewer-a", "channel_zyx": 2, "pair_left_zyx": [1, 2, 2]},
        {"decision": "DIFFERENT_PROCESS", "reviewer": "reviewer-a", "channel_zyx": 1, "pair_left_zyx": [0 if different_at_edge else 1, 2, 3]},
    ]
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    receipt = {"status":"REVIEWED_DVID_AFFINITY_SUPERVISION", "crop_id":region["id"], "split":region["role"], "raw_sha256":region["raw_sha256"],
        "targets":{"path":str(targets),"sha256":sha256_file(targets),"shape_czyx":[3,4,5,6]}, "mask":{"path":str(mask),"sha256":sha256_file(mask)},
        "event_log":{"path":str(log),"sha256":sha256_file(log)}, "effective_decisions":events}
    (out / "receipt.json").write_text(json.dumps(receipt))


def test_freeze_requires_interior_same_and_different(tmp_path: Path) -> None:
    ids = ["MV-G2-TRAIN-A", "MV-G2-TRAIN-B", "MV-G2-TRAIN-C", "MV-G2-TRAIN-D", "MV-G2-VALIDATION-V1", "MV-G2-VALIDATION-V2", "MV-G2-VALIDATION-V3"]
    regions=[]
    for i, identifier in enumerate(ids):
        role="G2_TARGET_TRAIN" if "TRAIN" in identifier else "G2_TARGET_VALIDATION"
        raw=tmp_path / f"{identifier}.npy"; np.save(raw, np.zeros((4,5,6), dtype=np.uint8))
        regions.append({"id":identifier,"role":role,"raw_path":str(raw),"raw_sha256":sha256_file(raw),"shape_zyx":[4,5,6],"bounds_xyz":{"x":[i*10,i*10+1],"y":[0,1],"z":[0,1]}})
    cohort=tmp_path / "cohort.json"; cohort.write_text(json.dumps({"regions":regions,"leave_one_region_out":[],"protected":"absent"}))
    receipts=tmp_path / "receipts"
    for r in regions: _receipt(receipts, r)
    output=tmp_path / "dataset.json"
    result=freeze_g2_supervision(cohort_path=cohort, receipt_root=receipts, output_path=output)
    assert result["status"] == "FROZEN_READY_FOR_G2_LORO"
    assert all(r["usable_counts"]["DIFFERENT_PROCESS"] == 1 for r in result["regions"])


def test_freeze_rejects_when_edge_filter_removes_different(tmp_path: Path) -> None:
    # The detailed success test covers structure; an edge-only DIFFERENT must fail closed.
    ids = ["MV-G2-TRAIN-A", "MV-G2-TRAIN-B", "MV-G2-TRAIN-C", "MV-G2-TRAIN-D", "MV-G2-VALIDATION-V1", "MV-G2-VALIDATION-V2", "MV-G2-VALIDATION-V3"]
    regions=[]
    for i, identifier in enumerate(ids):
        role="G2_TARGET_TRAIN" if "TRAIN" in identifier else "G2_TARGET_VALIDATION"; raw=tmp_path / f"{identifier}.npy"; np.save(raw,np.zeros((4,5,6),dtype=np.uint8))
        regions.append({"id":identifier,"role":role,"raw_path":str(raw),"raw_sha256":sha256_file(raw),"shape_zyx":[4,5,6],"bounds_xyz":{"x":[i*10,i*10+1],"y":[0,1],"z":[0,1]}})
    cohort=tmp_path / "cohort.json"; cohort.write_text(json.dumps({"regions":regions,"leave_one_region_out":[]})); receipts=tmp_path / "receipts"
    for r in regions: _receipt(receipts,r,different_at_edge=r["id"] == "MV-G2-TRAIN-A")
    with pytest.raises(ValueError, match="insufficient"):
        freeze_g2_supervision(cohort_path=cohort, receipt_root=receipts, output_path=tmp_path / "dataset.json")
