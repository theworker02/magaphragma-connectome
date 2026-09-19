"""Build G3_AXIS_NEUTRAL_SAMPLING_PROTOCOL_002 (supersedes -001; preserves it).

Controlled paired-location experiment: at each selected physical center (z,y,x)
the Z, Y and X adjacent-voxel edges are ALL eligible and separately reviewable,
so axis varies while the local 3-D tissue neighborhood is held constant.

Design (predeclared, fixed before any label):
  * Location pool = survey-eligible sources NOT already used by an existing
    review region (A-L). Label-blind.
  * Contrast strata: derived from RAW-ONLY pooled quantiles of interior
    per-center edge contrast (mean |c->c+1| over Z,Y,X) across ALL pooled
    eligible crops. 4 bands at pooled quartiles [0,25,50,75,100]%. Boundaries
    frozen BEFORE queue generation. C/E review outcomes play no role.
  * 12 locations = 4 strata x 3 locations; 3 axis edges each => 36 decisions.
  * Multi-crop per stratum: the 3 locations in a stratum must come from >=2
    distinct crops, else the design gate FAILS (no silent relaxation).
  * argmax(|gradient|) is NOT used for edge eligibility anywhere.
  * Exclude any exact affinity edge already present in historical queues.
  * Deterministic.
"""
from __future__ import annotations

import glob
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SURVEY = REPO / "experiments/phase6e/MV-G3-NEW-REGION-SURVEY-001.json"
MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
OUT_PROTOCOL = REPO / "experiments/phase6e/G3_AXIS_NEUTRAL_SAMPLING_PROTOCOL_002.json"
OUT_MANIFEST = REPO / "experiments/phase6e/g3-axis-neutral-queues-006/pre-review-manifest.json"
MARGIN = 3
SEP2 = 12 ** 2
N_STRATA = 4
LOCS_PER_STRATUM = 3
N_LOCATIONS = N_STRATA * LOCS_PER_STRATUM


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as s:
        while b := s.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def edge_contrast_field(arr: np.ndarray):
    """Per interior center c: mean(|c->c+1|) over Z,Y,X. Returns (coords, values)
    for all centers where c+1 is in-bounds on every axis and center is interior."""
    z, y, x = arr.shape
    core = arr[MARGIN:z-MARGIN, MARGIN:y-MARGIN, MARGIN:x-MARGIN]
    ez = np.abs(arr[MARGIN+1:z-MARGIN+1, MARGIN:y-MARGIN, MARGIN:x-MARGIN] - core)
    ey = np.abs(arr[MARGIN:z-MARGIN, MARGIN+1:y-MARGIN+1, MARGIN:x-MARGIN] - core)
    ex = np.abs(arr[MARGIN:z-MARGIN, MARGIN:y-MARGIN, MARGIN+1:x-MARGIN+1] - core)
    step = (ez + ey + ex) / 3.0
    zz, yy, xx = np.mgrid[MARGIN:z-MARGIN, MARGIN:y-MARGIN, MARGIN:x-MARGIN]
    coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    return coords, step.ravel()


def historical_edges() -> set:
    edges = set()
    for qp in glob.glob(str(REPO / "experiments/phase6e/g3-interface-queues-*/*.json")):
        try:
            q = json.loads(Path(qp).read_text())
        except Exception:
            continue
        for question in q.get("questions", []):
            edges.add((tuple(question["pair_left_zyx"]), tuple(question["pair_right_zyx"]), int(question["channel_zyx"])))
    return edges


def main() -> None:
    if OUT_PROTOCOL.exists():
        raise SystemExit(f"Refusing to overwrite {OUT_PROTOCOL}")
    survey = json.loads(SURVEY.read_text())
    manifest = json.loads(MANIFEST.read_text())
    rec = {r["id"]: r for r in manifest["records"]}

    # Location pool: survey-eligible sources not already used by A-L review.
    used_005 = {"MV-DVID-RAW-SURVEY-013", "MV-DVID-RAW-SURVEY-009", "MV-DVID-RAW-SURVEY-003", "MV-DVID-RAW-SURVEY-007"}
    pool_sources = [sid for sid in survey["eligible_candidates"] if sid not in used_005]
    pool_sources.sort()

    # --- Step 1: pool raw edge-contrast across all pool crops; freeze bands ---
    per_crop = {}
    pooled = []
    print(f"pool sources: {len(pool_sources)}", flush=True)
    for sid in pool_sources:
        arr = np.asarray(np.load(rec[sid]["raw_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        coords, vals = edge_contrast_field(arr)
        per_crop[sid] = (coords, vals)
        pooled.append(vals)
        print(f"  contrast field {sid}: {vals.size} centers", flush=True)
    pooled_all = np.concatenate(pooled)
    quartiles = np.quantile(pooled_all, [0.0, 0.25, 0.5, 0.75, 1.0])
    bands = []
    labels = ["q1_lowest", "q2_low_mid", "q3_mid_high", "q4_highest"]
    for i, label in enumerate(labels):
        bands.append({"label": label, "lo": float(quartiles[i]), "hi": float(quartiles[i+1])})

    def band_of(value: float) -> int:
        for i in range(N_STRATA):
            lo, hi = bands[i]["lo"], bands[i]["hi"]
            if (value >= lo and value < hi) or (i == N_STRATA - 1 and value <= hi):
                return i
        return N_STRATA - 1

    hist = historical_edges()
    print(f"historical edges excluded: {len(hist)}; bands: {[(b['label'], round(b['lo'],2), round(b['hi'],2)) for b in bands]}", flush=True)

    # --- Step 2: deterministic stratified location selection --------------
    # For each crop, get its greedy 12-separated interior centers (deterministic
    # by descending contrast then coord), tag each by band. Then fill strata.
    def separated_centers(coords, vals):
        order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], -vals))  # stable: high contrast first, then coord
        sep = 12
        grid = {}
        kept = []
        for idx in order:
            z, yy, xx = int(coords[idx][0]), int(coords[idx][1]), int(coords[idx][2])
            cell = (z//sep, yy//sep, xx//sep)
            conflict = False
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        for (az, ay, ax) in grid.get((cell[0]+dz, cell[1]+dy, cell[2]+dx), ()):
                            if (az-z)**2+(ay-yy)**2+(ax-xx)**2 < SEP2:
                                conflict = True
                                break
                        if conflict: break
                    if conflict: break
                if conflict: break
            if not conflict:
                grid.setdefault(cell, []).append((z, yy, xx))
                kept.append(((z, yy, xx), float(vals[idx])))
        return kept

    # candidate centers grouped by band, each carrying its crop
    band_candidates = {i: [] for i in range(N_STRATA)}
    for sid in pool_sources:
        coords, vals = per_crop[sid]
        sc = separated_centers(coords, vals)
        print(f"  separated centers {sid}: {len(sc)}", flush=True)
        for centre, val in sc:
            # all three edges must be in-bounds (guaranteed by interior+c+1) and
            # none of the three exact edges may be historical
            edges = {
                "Z": (centre, (centre[0]+1, centre[1], centre[2]), 0),
                "Y": (centre, (centre[0], centre[1]+1, centre[2]), 1),
                "X": (centre, (centre[0], centre[1], centre[2]+1), 2),
            }
            if any((e[0], e[1], e[2]) in hist for e in edges.values()):
                continue
            band_candidates[band_of(val)].append({"source_id": sid, "centre": centre, "contrast": val, "edges": edges})

    # deterministic order within band: by (source_id, centre)
    selected = []
    for i in range(N_STRATA):
        cands = sorted(band_candidates[i], key=lambda c: (c["source_id"], c["centre"]))
        # enforce multi-crop: greedily pick spreading across crops
        picked = []
        used_crops = []
        # round-robin across distinct crops to guarantee >=2 crops per stratum
        by_crop = {}
        for c in cands:
            by_crop.setdefault(c["source_id"], []).append(c)
        crop_cycle = sorted(by_crop)
        ci = 0
        while len(picked) < LOCS_PER_STRATUM and any(by_crop.values()):
            src = crop_cycle[ci % len(crop_cycle)]
            if by_crop[src]:
                picked.append(by_crop[src].pop(0))
            ci += 1
            if ci > 10000:
                break
        if len(picked) < LOCS_PER_STRATUM:
            raise SystemExit(f"DESIGN GATE FAILED: stratum {labels[i]} has only {len(picked)} location(s); cannot fill {LOCS_PER_STRATUM}")
        distinct_crops = {p["source_id"] for p in picked}
        if len(distinct_crops) < 2:
            raise SystemExit(f"DESIGN GATE FAILED: stratum {labels[i]} drawn from a single crop {distinct_crops}; multi-crop required")
        for p in picked:
            selected.append({**p, "band": labels[i]})

    if len(selected) != N_LOCATIONS:
        raise SystemExit(f"DESIGN GATE FAILED: selected {len(selected)} != {N_LOCATIONS}")

    protocol = {
        "id": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-002",
        "schema_version": 1,
        "status": "PREDECLARED_FIXED_BEFORE_LABELS",
        "supersedes": {"id": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-001", "path": "experiments/phase6e/G3_AXIS_NEUTRAL_SAMPLING_PROTOCOL_001.json", "preserved": True, "reason": "-001 was 9 locations, contrast-unbalanced (low/very_low only), single-crop-per-stratum; -002 uses raw-only pooled quartile strata, 12 locations, multi-crop per stratum."},
        "created_at": _now(),
        "band_derivation": {
            "algorithm": "pooled raw interior per-center edge contrast (mean |c->c+1| over Z,Y,X) across ALL pool crops; band edges at pooled quantiles [0,25,50,75,100]%",
            "label_blind": True,
            "pool_sources": pool_sources,
            "pooled_voxels": int(pooled_all.size),
            "frozen_before_queue": True,
            "ce_outcomes_used": False,
        },
        "contrast_bands": bands,
        "design": {"n_locations": N_LOCATIONS, "strata": N_STRATA, "locations_per_stratum": LOCS_PER_STRATUM, "axes_per_location": 3, "total_edges": N_LOCATIONS * 3, "multi_crop_per_stratum_required": True},
        "edge_definition": {"Z": "(z,y,x)<->(z+1,y,x)", "Y": "(z,y,x)<->(z,y+1,x)", "X": "(z,y,x)<->(z,y,x+1)"},
        "edge_eligibility": "all three edges eligible independently at each center; argmax(|gradient|) NOT used",
        "forbidden_inputs": ["SAME/DIFFERENT labels", "class predictions", "learned affinities", "historical class composition", "argmax-gradient edge eligibility"],
        "historical_edges_excluded": len(hist),
        "locations": [
            {
                "location_index": i + 1,
                "source_id": s["source_id"],
                "center_zyx": list(s["centre"]),
                "contrast_stratum": s["band"],
                "raw_edge_contrast": s["contrast"],
                "edges": {ax: {"pair_left_zyx": list(e[0]), "pair_right_zyx": list(e[1]), "channel_zyx": e[2]} for ax, e in s["edges"].items()},
            }
            for i, s in enumerate(selected)
        ],
    }
    OUT_PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol": str(OUT_PROTOCOL.resolve()),
        "bands": bands,
        "locations": len(selected),
        "per_stratum_crops": {labels[i]: sorted({s["source_id"] for s in selected if s["band"] == labels[i]}) for i in range(N_STRATA)},
    }, indent=2))


if __name__ == "__main__":
    main()
