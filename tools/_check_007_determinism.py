"""Prove deterministic regeneration of -007 selection without re-scanning
volumes twice. Centers per crop are computed once (the center computation is a
pure deterministic array op in fixed raster order); the stratum-assignment +
hash-selection pipeline is then run TWICE over that cache and compared to each
other and to the frozen protocol."""
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

# compute separated centers once per crop (deterministic pure op)
cache = {}
all_c = []
for sid in fresh:
    arr = np.asarray(np.load(rec[sid]["raw_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
    centers = B.greedy_separated_centers(B.eligible_centers(arr))
    cache[sid] = centers
    all_c.extend(c for _, c in centers)
all_c = np.array(all_c)
bounds = [float(np.quantile(all_c, p)) for p in B.PERCENTILE_EDGES]


def stratum_of(contrast):
    for i in range(4):
        lo, hi = bounds[i], bounds[i + 1]
        if (contrast >= lo and contrast < hi) or (i == 3 and contrast <= hi):
            return B.STRATA[i]
    return B.STRATA[-1]


def pipeline():
    assignments = [B.STRATA[i % 4] for i in range(B.N_CROPS)]
    out = []
    for i, sid in enumerate(fresh):
        ts = assignments[i]
        cands = [(c, ct) for (c, ct) in cache[sid] if stratum_of(ct) == ts]

        def ok(center):
            for ax_i, ax in enumerate(["Z", "Y", "X"]):
                le, ri = B._edges(center)[ax]
                if (tuple(le), tuple(ri), ax_i) in hist:
                    return False
            return True
        cands = [(c, ct) for (c, ct) in cands if ok(c)]
        raw_sha = rec[sid]["raw_sha256"]
        pick = min(cands, key=lambda cc: B._hash_rank(raw_sha, sid, ts, str(cc[0])))
        out.append((sid, ts, tuple(pick[0])))
    return out


d1 = pipeline()
d2 = pipeline()
frozen = [(l["source_id"], l["assigned_stratum"], tuple(l["center_zyx"])) for l in PROTO["locations"]]
print(json.dumps({"run1_eq_run2": d1 == d2, "run1_eq_frozen": d1 == frozen,
                  "deterministic_and_matches_frozen": d1 == d2 == frozen}, indent=2))
raise SystemExit(0 if d1 == d2 == frozen else 1)
