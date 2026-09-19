"""Investigate WHY regenerated -007 selection diverges from the frozen protocol.
Read-only. Compares, per crop: eligible-center count, contrast population,
percentile boundaries, assigned stratum, candidate count, and final pick."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_g3_007_axis_neutral_paired as B

REPO = Path(__file__).resolve().parents[1]
PROTO = json.loads((REPO / "experiments/phase6e/G3_AXIS_NEUTRAL_SAMPLING_PROTOCOL_003.json").read_text())
survey = json.loads(B.SURVEY.read_text())
manifest = json.loads(B.MANIFEST.read_text())
rec = {r["id"]: r for r in manifest["records"]}

fresh = sorted(s for s in survey["eligible_candidates"] if s not in B.USED_005 and s not in B.USED_006)[:B.N_CROPS]
hist = B.historical_edges()

print("fresh sources:", fresh)
print("frozen protocol sources:", [l["source_id"] for l in PROTO["locations"]])
print("frozen historical_edges_excluded_count:", PROTO.get("historical_edges_excluded_count"))
print("regenerated historical edge count:", len(hist))
print("frozen population boundaries:", PROTO["contrast_strata"]["population_contrast_boundaries"])

cache = {}
all_c = []
for sid in fresh:
    arr = np.asarray(np.load(rec[sid]["raw_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
    centers = B.greedy_separated_centers(B.eligible_centers(arr))
    cache[sid] = centers
    all_c.extend(c for _, c in centers)
all_c = np.array(all_c)
bounds = [float(np.quantile(all_c, p)) for p in B.PERCENTILE_EDGES]
print("regenerated population boundaries:", bounds)
print("regenerated center counts per crop:", {sid: len(cache[sid]) for sid in fresh})

frozen_by_source = {l["source_id"]: l for l in PROTO["locations"]}
for i, sid in enumerate(fresh):
    fl = frozen_by_source.get(sid)
    print(f"  {sid}: frozen_center={fl['center_zyx'] if fl else None} frozen_stratum={fl['assigned_stratum'] if fl else None} frozen_contrast={fl['raw_contrast'] if fl else None}")
