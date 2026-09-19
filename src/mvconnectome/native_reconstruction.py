"""DVID-native, machine-only reconstruction baseline.

This is deliberately identified as marker-controlled watershed, never FFN.
It operates only on the immutable local DVID crop and produces candidate
segments/fragments, not biological neuron identities.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from .io import sha256_file, write_json_atomic

BACKEND = "MV-SEG-BACKEND-WATERSHED-0001"

def run_first_chunk(ingestion_path: Path, chunks_path: Path, output: Path) -> dict:
    """Run one marker-controlled baseline chunk and retain machine-only receipts."""
    ingestion = json.loads(ingestion_path.read_text())
    chunks_doc = json.loads(chunks_path.read_text()); chunk = chunks_doc["chunks"][0]
    origin = ingestion["source_bounds_xyz"]; read = chunk["read_bounds_xyz"]
    raw = np.load(ingestion["raw_array"], mmap_mode="r")
    sl = (slice(read["z"][0]-origin["z"][0], read["z"][1]-origin["z"][0]), slice(read["y"][0]-origin["y"][0], read["y"][1]-origin["y"][0]), slice(read["x"][0]-origin["x"][0], read["x"][1]-origin["x"][0]))
    image = np.asarray(raw[sl], dtype=np.float32); started=time.time()
    lo, hi = np.percentile(image, (1,99)); norm=np.clip((image-lo)/max(hi-lo,1e-6),0,1)
    smooth=ndimage.gaussian_filter(norm, sigma=(0.8,1,1)); grad=np.sqrt(sum(c*c for c in np.gradient(smooth)))
    membrane=np.clip(grad/max(float(grad.max()),1e-9)*255,0,255).astype(np.uint8)
    # Candidate seeds are deterministic low-gradient interior locations; this is a
    # machine proposal, not a biological interior classification.
    markers=np.zeros(image.shape,dtype=np.int32); serial=1; threshold=np.percentile(membrane,60)
    for z in range(4,image.shape[0]-4,8):
      for y in range(24,image.shape[1]-24,48):
       for x in range(24,image.shape[2]-24,48):
        if membrane[z,y,x] <= threshold: markers[z,y,x]=serial; serial+=1
    if serial == 1: raise RuntimeError("No DVID-native low-gradient candidate seed survived threshold")
    labels=ndimage.watershed_ift(membrane,markers)
    seed_voxel=np.argwhere(markers>0)[0]; seed_label=int(labels[tuple(seed_voxel)])
    mask=labels==seed_label; positions=np.argwhere(mask)
    bbox_lo,bbox_hi=positions.min(0),positions.max(0)+1
    boundary=any(bbox_lo[i]==0 or bbox_hi[i]==mask.shape[i] for i in range(3))
    run_dir=output/chunk["id"]/"run-MV-DVID-NATIVE-000001"; run_dir.mkdir(parents=True,exist_ok=True)
    np.save(run_dir/"membrane_probability_u8.npy",membrane); np.save(run_dir/"labels_zyx.npy",labels); np.save(run_dir/"seed_mask_zyx.npy",markers)
    global_seed=[read["x"][0]+int(seed_voxel[2]),read["y"][0]+int(seed_voxel[1]),read["z"][0]+int(seed_voxel[0])]
    seg={"id":"MV-SEG-DVID-00000001","origin":"DVID_NATIVE","specimen_id":"SPECIMEN_UNKNOWN","status":"MACHINE_SEGMENTED","review_state":"MACHINE_ONLY","volume_id":ingestion["volume_id"],"chunk_id":chunk["id"],"backend_id":BACKEND,"run_id":"MV-DVID-NATIVE-RUN-000001","seed_id":"MV-SEED-DVID-00000001","voxel_bounds_xyz":{"x":[read["x"][0]+int(bbox_lo[2]),read["x"][0]+int(bbox_hi[2])],"y":[read["y"][0]+int(bbox_lo[1]),read["y"][0]+int(bbox_hi[1])],"z":[read["z"][0]+int(bbox_lo[0]),read["z"][0]+int(bbox_hi[0])]},"voxel_count":int(mask.sum()),"input_raw_sha256":ingestion["raw_sha256"],"labels_sha256":sha256_file(run_dir/"labels_zyx.npy"),"boundary_contact":boundary}
    seed={"id":"MV-SEED-DVID-00000001","origin":"DVID_NATIVE","specimen_id":"SPECIMEN_UNKNOWN","volume_id":ingestion["volume_id"],"chunk_id":chunk["id"],"voxel_xyz":global_seed,"physical_xyz_nm":[v*8 for v in global_seed],"method":"low-gradient interior candidate from real DVID EM","status":"USED","confidence":None,"warning":"Automated interior proposal; not biological ground truth."}
    fragment={"id":"MV-FRAG-DVID-00000001","origin":"DVID_NATIVE","specimen_id":"SPECIMEN_UNKNOWN","segment_ids":[seg["id"]],"status":"MACHINE_ASSEMBLED","volume_id":ingestion["volume_id"],"chunk_ids":[chunk["id"]],"seed_id":seed["id"],"backend_run":seg["run_id"],"boundary_status":"CHUNK_BOUNDARY" if boundary else "UNTRACED_CONTINUATION","skeleton_status":"BLOCKED_NO_3D_SKELETONIZER","evidence":{"raw_sha256":ingestion["raw_sha256"],"label_sha256":seg["labels_sha256"]}}
    receipt={"id":seg["run_id"],"backend":{"id":BACKEND,"name":"gradient-derived membrane proxy + marker-controlled watershed","is_ffn":False,"framework":"SciPy ndimage","model":None,"model_hash":None},"status":"SUCCEEDED","input":{"raw_sha256":ingestion["raw_sha256"],"read_bounds_xyz":read,"shape_zyx":list(image.shape)},"runtime_seconds":time.time()-started,"hardware":{"backend":"CPU","platform":platform.platform()},"outputs":{"labels_sha256":seg["labels_sha256"]},"release_eligible":False}
    for name,value in [("seed.json",seed),("segments.json",{"segments":[seg]}),("fragments.json",{"fragments":[fragment]}),("run.json",receipt)]: write_json_atomic(run_dir/name,value)
    chunk["processing_state"]="SEGMENTED"; chunk["segmentation_state"]="MACHINE_SEGMENTED"; chunk["skeleton_state"]="BLOCKED"; chunks_path.write_text(json.dumps(chunks_doc,indent=2)+"\n")
    return {"run_dir":str(run_dir),"seed":seed["id"],"segment":seg["id"],"fragment":fragment["id"],"voxel_count":seg["voxel_count"],"boundary_contact":boundary}
