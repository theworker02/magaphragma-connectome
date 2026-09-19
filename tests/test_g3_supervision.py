import json
from pathlib import Path

import numpy as np
import pytest

from mvconnectome.g3_supervision import freeze_g3_supervision
from mvconnectome.io import sha256_file


def _write_region(root: Path, region: dict, *, mixed: bool = False) -> dict:
    raw = root / f"{region['id']}.npy"; np.save(raw, np.zeros((6, 7, 8), dtype=np.uint8))
    workspace = root / f"{region['id']}-workspace.json"; log = root / f"{region['id']}.events.jsonl"; queue = root / f"{region['id']}-queue.json"
    questions=[]; events=[]
    for index, decision in enumerate(("SAME_PROCESS", "DIFFERENT_PROCESS")):
        iid=f"I-{index}"; left=[2, 2 + index, 2]
        questions.append({"id":f"Q-{index}","interface_id":iid,"pair_left_zyx":left,"pair_right_zyx":[left[0], left[1], left[2]+1],"channel_zyx":2})
        actual="DIFFERENT_PROCESS" if mixed and index == 0 else decision
        events.append({"crop_id":region["id"],"split":region["role"],"raw_sha256":sha256_file(raw),"reviewer":"expert","decision":actual,"question_reference":f"Q-{index}","interface_id":iid,"pair_left_zyx":left,"pair_right_zyx":[left[0], left[1], left[2]+1],"channel_zyx":2})
    log.write_text("\n".join(json.dumps(item) for item in events) + "\n")
    workspace.write_text(json.dumps({"id":f"W-{region['id']}","crop_id":region["id"],"split":region["role"],"parent_region_id":region["source_id"],"raw":{"path":str(raw),"sha256":sha256_file(raw),"shape_zyx":[6,7,8]},"event_log":{"path":str(log)}}))
    queue.write_text(json.dumps({"status":"EXPERT_INTERFACE_REVIEW_REQUIRED","crop_id":region["id"],"workspace_id":f"W-{region['id']}","raw_sha256":sha256_file(raw),"questions":questions}))
    region = {**region,"raw_path":str(raw),"raw_sha256":sha256_file(raw),"shape_zyx":[6,7,8]}
    return {"region":region,"workspace":str(workspace),"queue":str(queue)}


def test_freeze_g3_normalizes_each_interface_and_rejects_missing_class(tmp_path: Path) -> None:
    regions=[]; bindings={}
    for index in range(9):
        role="G3_TARGET_TRAIN" if index < 6 else "G3_TARGET_VALIDATION"
        seed={"id":f"R{index}","role":role,"source_id":f"S{index}","bounds_xyz":{"x":[index*10,index*10+1],"y":[0,1],"z":[0,1]}}
        built=_write_region(tmp_path, seed); regions.append(built["region"]); bindings[seed["id"]]={"workspace":built["workspace"],"queue":built["queue"]}
    cohort=tmp_path/"cohort.json"; cohort.write_text(json.dumps({"regions":regions,"loro_folds":[]})); bound=tmp_path/"bindings.json"; bound.write_text(json.dumps({"regions":bindings}))
    result=freeze_g3_supervision(cohort_path=cohort,inputs_path=bound,output_root=tmp_path/"out",manifest_path=tmp_path/"manifest.json")
    assert result["status"] == "FROZEN_READY_FOR_G3_LORO"
    weights=np.load(tmp_path/"out"/"R0"/"interface_weights_czyx.npy")
    assert weights.sum() == pytest.approx(2.0)


def test_freeze_g3_fails_when_region_loses_same_class(tmp_path: Path) -> None:
    regions=[]; bindings={}
    for index in range(9):
        role="G3_TARGET_TRAIN" if index < 6 else "G3_TARGET_VALIDATION"
        seed={"id":f"R{index}","role":role,"source_id":f"S{index}","bounds_xyz":{"x":[index*10,index*10+1],"y":[0,1],"z":[0,1]}}
        built=_write_region(tmp_path, seed, mixed=index == 0); regions.append(built["region"]); bindings[seed["id"]]={"workspace":built["workspace"],"queue":built["queue"]}
    cohort=tmp_path/"cohort.json"; cohort.write_text(json.dumps({"regions":regions,"loro_folds":[]})); bound=tmp_path/"bindings.json"; bound.write_text(json.dumps({"regions":bindings}))
    with pytest.raises(ValueError, match="SAME/DIFFERENT"):
        freeze_g3_supervision(cohort_path=cohort,inputs_path=bound,output_root=tmp_path/"out",manifest_path=tmp_path/"manifest.json")
