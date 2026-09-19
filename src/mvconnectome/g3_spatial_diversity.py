"""Freeze a spatially and raw-appearance diverse G3 supervision cohort."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file, write_json_atomic


def _now() -> str: return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(max(a["bounds_xyz"][axis][0], b["bounds_xyz"][axis][0]) < min(a["bounds_xyz"][axis][1], b["bounds_xyz"][axis][1]) for axis in ("x", "y", "z"))


def freeze_g3_cohort(*, survey_path: Path, plan_path: Path, output_path: Path) -> dict[str, Any]:
    # The cohort is frozen before any expert answer or model prediction can
    # influence it.  A later replacement must create a new cohort version.
    if output_path.exists(): raise FileExistsError(f"Refusing to overwrite frozen G3 cohort: {output_path}")
    survey, plan = json.loads(survey_path.read_text()), json.loads(plan_path.read_text())
    index = {record["id"]: record for record in survey["records"]}; selected = plan["selected_regions"]
    if len(selected) != 9 or sum(item["role"] == "G3_TARGET_TRAIN" for item in selected) != 6 or sum(item["role"] == "G3_TARGET_VALIDATION" for item in selected) != 3:
        raise ValueError("G3 requires exactly six TRAIN and three independent VALIDATION regions")
    records=[]
    for item in selected:
        # Recompute the stored raw statistics from the immutable crop instead
        # of trusting the plan; these describe sampling coverage, not labels.
        source=index.get(item["source_id"])
        if source is None or item["source_id"] in plan["excluded_sources"]: raise ValueError(f"Invalid or excluded G3 source: {item['source_id']}")
        raw=Path(source["raw_path"])
        if not raw.is_file() or sha256_file(raw) != source["raw_sha256"]: raise ValueError(f"Raw/hash failure: {item['source_id']}")
        array=np.load(raw,mmap_mode="r"); q=np.quantile(array,[.01,.1,.5,.9,.99]).astype(float).tolist()
        gradient=float(sum(np.mean(np.abs(np.diff(array.astype(np.float32),axis=axis))) for axis in range(3)))
        records.append({"id":item["id"],"role":item["role"],"source_id":item["source_id"],"raw_stratum":item["raw_stratum"],"bounds_xyz":source["bounds_xyz"],"raw_path":str(raw.resolve()),"raw_sha256":source["raw_sha256"],"shape_zyx":source["shape_zyx"],"dtype":source["dtype"],"voxel_size_nm_xyz":survey["source"]["voxel_size_nm_xyz"],"pre_review_raw_statistics":{"mean":float(array.mean()),"std":float(array.std()),"quantiles_p01_p10_p50_p90_p99":q,"mean_abs_3d_gradient":gradient}})
    if len({record["id"] for record in records}) != len(records): raise ValueError("G3 IDs must be unique")
    # Spatial disjointness protects TRAIN/VALIDATION and LORO comparisons from
    # voxel leakage between regions.
    for i,left in enumerate(records):
        for right in records[i+1:]:
            if _overlap(left,right): raise ValueError(f"G3 spatial overlap: {left['id']} / {right['id']}")
    train=[r["id"] for r in records if r["role"] == "G3_TARGET_TRAIN"]
    result={"schema_version":1,"id":plan["id"],"created_at":_now(),"status":"FROZEN_PENDING_G3_INTERFACE_REVIEW","survey":{"path":str(survey_path.resolve()),"sha256":sha256_file(survey_path)},"plan":{"path":str(plan_path.resolve()),"sha256":sha256_file(plan_path)},"selection":{"method":"SPATIAL_PLUS_PRE_REVIEW_RAW_APPEARANCE_STRATIFICATION_V1","prohibited_inputs":plan["prohibited_inputs"]},"regions":records,"loro_folds":[{"held_out_region":held,"train_with":[x for x in train if x != held]} for held in train],"protected_regression":{"volume":"MV-GTVOL-000004","included":False},"review_requirement":"Each region requires reviewed SAME_PROCESS, DIFFERENT_PROCESS, and a boundary-interface decision package before G3 training."}
    write_json_atomic(output_path,result); return result


def create_g3_review_inputs(*, cohort_path: Path, output_dir: Path) -> dict[str, Any]:
    """Create raw-only G3 review manifests; no model result selects a crop."""
    if output_dir.exists(): raise FileExistsError(f"Refusing to overwrite G3 review inputs: {output_dir}")
    cohort=json.loads(cohort_path.read_text())
    if cohort.get("status") != "FROZEN_PENDING_G3_INTERFACE_REVIEW": raise ValueError("G3 cohort is not frozen for review")
    output_dir.mkdir(parents=True)
    request=[]; crops=[]
    for region in cohort["regions"]:
        # The raw hash is checked again immediately before a reviewer package
        # is created, preserving a direct raw-voxel-to-decision lineage.
        raw=Path(region["raw_path"])
        if not raw.is_file() or sha256_file(raw) != region["raw_sha256"]: raise ValueError(f"Raw/hash mismatch: {region['id']}")
        origin=[region["bounds_xyz"][axis][0] for axis in ("x","y","z")]
        request.append({"crop_id":region["id"],"parent_region_id":region["source_id"],"raw_sha256":region["raw_sha256"],"coordinate_frame":"MV-FRAME-DVID-WASP5-001","source_origin_xyz":origin,"role":region["role"]})
        crops.append({"id":region["id"],"parent_region_id":region["source_id"],"raw_crop_path":str(raw.resolve()),"raw_crop_sha256":region["raw_sha256"],"raw_shape_zyx":region["shape_zyx"],"raw_dtype":region["dtype"],"coordinate_frame":"MV-FRAME-DVID-WASP5-001","source_origin_xyz":origin,"bounds_xyz":region["bounds_xyz"],"split":region["role"],"status":"PENDING_REVIEWED_PAIR_AND_INTERFACE_EVIDENCE"})
    request_path=output_dir / "request.json"; manifest_path=output_dir / "crop-manifest.json"
    write_json_atomic(request_path,{"schema_version":1,"id":"MV-G3-EXTERNAL-REVIEW-REQUEST-001","cohort":{"path":str(cohort_path.resolve()),"sha256":sha256_file(cohort_path)},"requested_review_sets":request,"prohibited_sources":["MV-GTVOL-000004"]})
    write_json_atomic(manifest_path,{"schema_version":1,"id":"MV-G3-EXTERNAL-REVIEW-CROPS-001","cohort":{"path":str(cohort_path.resolve()),"sha256":sha256_file(cohort_path)},"crops":crops,"prohibited_sources":["MV-GTVOL-000004"]})
    return {"request":str(request_path.resolve()),"manifest":str(manifest_path.resolve()),"regions":len(crops)}
