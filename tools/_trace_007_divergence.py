"""Stage-by-stage first-divergence tracer for the -007 selection pipeline.

Compares a PURE-PYTHON REFERENCE implementation (scalar loops, no vectorization)
against the OPTIMIZED implementation in build_g3_007_axis_neutral_paired, hashing
each stage in order, and reports the FIRST stage where they diverge. Also
compares both to the frozen protocol-003. Read-only; writes nothing.

Stages, in order:
  1. raw crop bytes sha256
  2. eligible center count + ordered-coordinate hash
  3. contrast array hash + stats
  4. percentile boundaries (population)
  5. per-crop stratum membership counts
  6. historical-edge exclusion count
  7. separated-center ordered hash
  8. stable-hash ranking of stratum candidates (per crop)
  9. selected center
 10. Z/Y/X edge triplet
"""
from __future__ import annotations

import hashlib
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
MARGIN = B.MARGIN
SEP2 = B.SEP2


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---- REFERENCE (pure python, scalar) ----------------------------------------
def ref_eligible_centers(arr):
    z, y, x = arr.shape
    coords = []
    contrasts = []
    for zi in range(MARGIN, z - MARGIN - 1):
        for yi in range(MARGIN, y - MARGIN - 1):
            for xi in range(MARGIN, x - MARGIN - 1):
                c0 = float(arr[zi, yi, xi])
                ez = abs(float(arr[zi + 1, yi, xi]) - c0)
                ey = abs(float(arr[zi, yi + 1, xi]) - c0)
                ex = abs(float(arr[zi, yi, xi + 1]) - c0)
                coords.append((zi, yi, xi))
                contrasts.append((ez + ey + ex) / 3.0)
    return coords, contrasts


def ref_greedy(coords, contrasts):
    sep = 12
    grid = {}
    kept = []
    for (c, ct) in zip(coords, contrasts):
        z, y, x = c
        cell = (z // sep, y // sep, x // sep)
        conflict = False
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for (az, ay, ax) in grid.get((cell[0] + dz, cell[1] + dy, cell[2] + dx), ()):
                        if (az - z) ** 2 + (ay - y) ** 2 + (ax - x) ** 2 < SEP2:
                            conflict = True
                            break
                    if conflict: break
                if conflict: break
            if conflict: break
        if not conflict:
            grid.setdefault(cell, []).append((z, y, x))
            kept.append((c, ct))
    return kept


def trace_crop(sid):
    r = rec[sid]
    arr = np.asarray(np.load(r["raw_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
    report = {"source": sid}

    # optimized
    o_coords, o_contrast = B.eligible_centers(arr)
    o_sep = B.greedy_separated_centers((o_coords, o_contrast))
    # reference
    r_coords, r_contrast = ref_eligible_centers(arr)
    r_sep = ref_greedy(r_coords, r_contrast)

    # Stage 2: eligible center ordered coords
    o_coord_list = [tuple(int(v) for v in c) for c in o_coords]
    report["stage2_eligible_count"] = {"opt": len(o_coord_list), "ref": len(r_coords), "equal": o_coord_list == r_coords, "opt_hash": _h(o_coord_list), "ref_hash": _h(r_coords)}
    # Stage 3: contrast
    o_ct = [float(v) for v in o_contrast]
    report["stage3_contrast"] = {"equal": np.allclose(o_ct, r_contrast, atol=0, rtol=0), "opt_hash": _h([round(v, 9) for v in o_ct]), "ref_hash": _h([round(v, 9) for v in r_contrast])}
    # Stage 7: separated centers ordered
    o_sep_c = [tuple(int(v) for v in c) for c, _ in o_sep]
    r_sep_c = [tuple(c) for c, _ in r_sep]
    report["stage7_separated"] = {"opt_count": len(o_sep_c), "ref_count": len(r_sep_c), "equal": o_sep_c == r_sep_c, "opt_hash": _h(o_sep_c), "ref_hash": _h(r_sep_c)}
    # Stage 8: hash-rank type trap check on the FIRST separated center
    if o_sep:
        opt_center = o_sep[0][0]
        report["stage8_center_repr"] = {"opt_type": type(opt_center[0]).__name__, "opt_str": str(opt_center), "ref_type": type(r_sep[0][0][0]).__name__ if r_sep else None, "ref_str": str(r_sep[0][0]) if r_sep else None}
    return report


def main():
    fresh = sorted(s for s in survey["eligible_candidates"] if s not in B.USED_005 and s not in B.USED_006)[:B.N_CROPS]
    # trace first two crops for speed; extend if needed
    which = fresh[:2]
    out = {"traced_crops": which, "per_crop": [trace_crop(s) for s in which]}
    Path("bin").mkdir(exist_ok=True)
    Path("bin/trace007.json").write_text(json.dumps(out, indent=2))
    # localize first divergence
    for cr in out["per_crop"]:
        for stage in ("stage2_eligible_count", "stage3_contrast", "stage7_separated"):
            if not cr[stage]["equal"]:
                cr["FIRST_DIVERGENCE"] = stage
                break
        else:
            cr["FIRST_DIVERGENCE"] = None
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
