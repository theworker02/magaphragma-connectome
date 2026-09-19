"""Create a raw-EM-only, grouped local-interface review queue for frozen G3 crops.

Candidate selection uses raw-EM 3-D gradient geometry only. SegNeuron,
supervoxels, prior labels, reconstruction output, and reviewed SAME/DIFFERENT
labels are all excluded from selection. The optional paging/exclusion/
orientation controls below remain strictly label-blind: they operate on raw
gradient rank, prior *question geometry* (never prior answers), and the
physical axis a raw interface is oriented along. None of them consult, read,
or correlate with any SAME_PROCESS/DIFFERENT_PROCESS decision.
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import UTC, datetime
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__); p.add_argument("workspace",type=Path); p.add_argument("--output",type=Path,required=True); p.add_argument("--interfaces",type=int,default=8); p.add_argument("--mode",choices=("boundary","continuity"),default="boundary")
# --- Minimal, backward-compatible, label-blind selection extension ---------
# When none of these are supplied the generator behaves exactly as before.
p.add_argument("--exclude-queue",type=Path,action="append",default=[],help="Prior queue JSON whose exact interface pairs must be excluded (repeatable). Uses prior question GEOMETRY only, never prior answers.")
p.add_argument("--skip-candidates",type=int,default=0,help="Deterministically page past this many highest-ranked raw-EM interface centres before selecting, to reach fresh candidates.")
p.add_argument("--orientation-axis",choices=("Z","Y","X"),default=None,help="Restrict selection to raw interfaces geometrically oriented along this physical axis (0=Z,1=Y,2=X). Orientation is raw geometry, not a biological class.")
a=p.parse_args()
_AXIS_INDEX={"Z":0,"Y":1,"X":2}
# A queue is immutable once created: review questions cannot be re-ranked after
# their answers are known, and a later model result cannot overwrite it.
if a.output.exists(): raise FileExistsError(f"Refusing to overwrite frozen interface queue: {a.output}")
if a.interfaces<=0 or a.skip_candidates<0: raise ValueError("interfaces must be positive and skip-candidates non-negative")
w=json.loads(a.workspace.read_text()); raw=np.load(w["raw"]["path"],mmap_mode="r",allow_pickle=False)
if hashlib.sha256(Path(w["raw"]["path"]).read_bytes()).hexdigest()!=w["raw"]["sha256"]: raise ValueError("Raw hash mismatch")

# Collect every exact (left,right,channel) pair already asked in prior queues so
# fresh review never repeats an interface. This reads prior question geometry
# only; prior SAME/DIFFERENT answers are never opened here.
excluded_pairs:set[tuple]=set(); exclusion_sources=[]
for queue_path in a.exclude_queue:
 prior=json.loads(queue_path.read_text())
 for question in prior.get("questions",[]):
  excluded_pairs.add((tuple(question["pair_left_zyx"]),tuple(question["pair_right_zyx"]),int(question["channel_zyx"])))
 exclusion_sources.append({"path":str(queue_path.resolve()),"sha256":hashlib.sha256(queue_path.read_bytes()).hexdigest(),"excluded_questions":len(prior.get("questions",[]))})

arr=np.asarray(raw,dtype=np.float32); grads=np.stack(np.gradient(arr)); score=np.linalg.norm(grads,axis=0); margin=3
# This ranking uses raw EM gradients only.  SegNeuron, supervoxels, prior
# labels, and reconstruction output are intentionally absent from selection.
order=np.argsort(score.ravel())[::-1 if a.mode == "boundary" else 1]; centres=[]; skipped=0
target_axis=_AXIS_INDEX[a.orientation_axis] if a.orientation_axis else None
for flat in order:
 z,y,x=map(int,np.unravel_index(flat,score.shape))
 if min(z,y,x) < margin or z>=arr.shape[0]-margin or y>=arr.shape[1]-margin or x>=arr.shape[2]-margin: continue
 axis=int(np.argmax(np.abs(grads[(slice(None),)+(z,y,x)])))
 # Orientation filter is raw geometry only: it selects which physical face a
 # candidate straddles, never whether it is SAME or DIFFERENT.
 if target_axis is not None and axis!=target_axis: continue
 # Reconstruct this candidate's member pairs to test against prior queues and
 # against already-accepted centres, so fresh review cannot duplicate a pair.
 other=[d for d in range(3) if d!=axis]
 offsets=((0,0),(-1,0),(1,0)) if a.mode == "boundary" else ((0,0),)
 member_pairs=[]
 for offset in offsets:
  left=[z,y,x]; left[other[0]]+=offset[0]; left[other[1]]+=offset[1]
  member_pairs.append((tuple(left),tuple(left[:axis]+[left[axis]+1]+left[axis+1:]),axis))
 if any((mp[0],mp[1],mp[2]) in excluded_pairs for mp in member_pairs): continue
 if all((z-c[0])**2+(y-c[1])**2+(x-c[2])**2 >= 12**2 for c in centres):
  # Deterministic paging: skip the highest-ranked eligible centres first so a
  # later tier can be reached reproducibly.
  if skipped<a.skip_candidates: skipped+=1; continue
  centres.append((z,y,x))
 if len(centres)>=a.interfaces: break
if len(centres)<a.interfaces: raise ValueError(f"Insufficient fresh spatially separated raw-EM interface candidates (found {len(centres)}, requested {a.interfaces}); relax orientation/skip or review a different crop")
questions=[]
for interface_index, centre in enumerate(centres,1):
 axis=int(np.argmax(np.abs(grads[(slice(None),)+centre])))
 other=[d for d in range(3) if d!=axis]
 # Boundary groups receive three paired observations; continuity controls are
 # deliberately one independent low-gradient local pair each.
 offsets=((0,0),(-1,0),(1,0)) if a.mode == "boundary" else ((0,0),)
 for member,offset in enumerate(offsets,1):
  left=list(centre); left[other[0]]+=offset[0]; left[other[1]]+=offset[1]
  kind="RAW_EM_INTERFACE_MEMBER" if a.mode == "boundary" else "RAW_EM_CONTINUITY_CONTROL"
  prefix="MV-G3-IF" if a.mode == "boundary" else "MV-G3-CONTROL"
  questions.append({"id":f"{prefix}-{interface_index:03d}-{member:02d}","kind":kind,"interface_id":f"{prefix}-{interface_index:03d}","interface_member":member,"interface_members":len(offsets),"pair_left_zyx":left,"pair_right_zyx":left[:axis]+[left[axis]+1]+left[axis+1:],"channel_zyx":axis,"raw_gradient_score":float(score[centre]),"selection":"RAW_EM_3D_GRADIENT_INTERFACE_GROUP_V1" if a.mode == "boundary" else "RAW_EM_LOW_GRADIENT_CONTINUITY_CONTROL_V1","model_navigation":False})
# An interface is merely a review unit.  It receives no biological label until
# an identified expert records an append-only decision for every member.
selection_record={"method":"RAW_EM_3D_GRADIENT_INTERFACE_GROUP_V1" if a.mode == "boundary" else "RAW_EM_LOW_GRADIENT_CONTINUITY_CONTROL_V1","interfaces":a.interfaces,"direct_pairs_per_interface":3 if a.mode == "boundary" else 1,"prohibited_inputs":["SegNeuron","G2 predictions","supervoxels","prior labels","reviewed SAME/DIFFERENT decisions"],"skip_candidates":a.skip_candidates,"orientation_axis":a.orientation_axis,"orientation_axis_channel_zyx":target_axis,"label_blind":True,"exclusion_sources":exclusion_sources,"excluded_pair_count":len(excluded_pairs)}
value={"schema_version":1,"id":f"MV-G3-RAW-{a.mode.upper()}-QUEUE-{w['crop_id'][-6:]}","created_at":datetime.now(UTC).isoformat().replace('+00:00','Z'),"status":"EXPERT_INTERFACE_REVIEW_REQUIRED","workspace_id":w["id"],"crop_id":w["crop_id"],"raw_sha256":w["raw"]["sha256"],"selection":selection_record,"questions":questions,"scientific_boundary":"Raw features choose review locations only. Each direct pair remains an independent expert decision; no group is automatically promoted."}
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(value,indent=2)+"\n")
print(json.dumps({"id":value['id'],"questions":len(questions),"interfaces":a.interfaces,"orientation_axis":a.orientation_axis,"skip_candidates":a.skip_candidates,"excluded_pairs":len(excluded_pairs),"output":str(a.output.resolve())}))
