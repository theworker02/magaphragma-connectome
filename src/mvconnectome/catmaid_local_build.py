"""Derive quarantined local source-connectome records from immutable CATMAID rows."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from .io import sha256_file, write_json_atomic

def build(rows_path: Path, output: Path) -> Path:
    """Derive quarantined source-connectome records without granting release status."""
    rows = json.loads(rows_path.read_text())
    by_connector = defaultdict(list)
    for row in rows: by_connector[str(row[0])].append(row)
    skeletons = sorted({int(row[4]) for row in rows})
    neurons = [{"id":f"MV-N-{i:06d}","source_system":"CATMAID","source_skeleton_id":sid,"status":"SOURCE_RECONSTRUCTED","morphology":"SOURCE_AVAILABLE_REMOTE_NOT_YET_CACHED","coordinate_frame":"MV-FRAME-CATMAID-001","specimen_id":"UNKNOWN","source_artifact_sha256":sha256_file(rows_path)} for i,sid in enumerate(skeletons,1)]
    mv_for = {n["source_skeleton_id"]:n["id"] for n in neurons}
    synapses=[]; connections=defaultdict(list); unresolved=[]
    for cid, links in by_connector.items():
        pre=[int(r[4]) for r in links if int(r[10])==15]; post=[int(r[4]) for r in links if int(r[10])==16]
        if not pre or not post:
            missing = "NO_PRE_LINK" if not pre else "NO_POST_LINK"
            unresolved.append({"connector_id":cid,"pre_skeletons":pre,"post_skeletons":post,
                               "missing":missing,"solvability":"RESOLVABLE_FROM_DOCUMENTED_SOURCE_QUERY"})
            continue
        syn_id=f"MV-SYN-{len(synapses)+1:08d}"; first=links[0]
        synapses.append({"id":syn_id,"source_connector_id":cid,"status":"SOURCE_ANNOTATED","coordinates_nm_xyz":first[1:4],"pre_neurons":[mv_for[x] for x in sorted(set(pre))],"post_neurons":[mv_for[x] for x in sorted(set(post))],"source_link_rows":len(links)})
        for a in set(pre):
            for b in set(post): connections[(mv_for[a],mv_for[b])].append(syn_id)
    # Source annotation is evidence status, not a project-review decision.
    conns=[{"id":f"MV-CONN-{i:08d}","pre_neuron_id":a,"post_neuron_id":b,"synapse_ids":v,"status":"SOURCE_ANNOTATED","review_state":"UNREVIEWED"} for i,((a,b),v) in enumerate(sorted(connections.items()),1)]
    output.mkdir(parents=True,exist_ok=True)
    write_json_atomic(output/'neurons.json',{"neurons":neurons}); write_json_atomic(output/'synapses.json',{"synapses":synapses}); write_json_atomic(output/'connections.json',{"connections":conns}); write_json_atomic(output/'unresolved_connectors.json',{"connectors":unresolved})
    write_json_atomic(output/'manifest.json',{"kind":"LOCAL_RESEARCH_BUILD","release_eligible":False,"source_artifact_sha256":sha256_file(rows_path),"counts":{"source_skeletons":len(skeletons),"neurons":len(neurons),"connectors":len(by_connector),"synapses":len(synapses),"connections":len(conns),"unresolved_connectors":len(unresolved)}})
    return output/'manifest.json'
