"""Build the -007 axis-neutral paired-location experiment (protocol-003).

Primary question: at the SAME fresh physical location, do the independently
reviewed Z, Y, X adjacent affinity edges retain the historical axis/class
relationship? Holding the local tissue constant removes crop/location
confounding.

Design (predeclared, label-blind, fixed before any label):
  * 12 distinct fresh eligible crops (canonical source-ID order), one physical
    location each.
  * relative within-eligible-population contrast strata Q1..Q4 by percentile
    rank of raw-only local edge contrast; boundaries frozen before selection.
  * fixed stratum rotation by crop order: Q1,Q2,Q3,Q4,Q1,... => 3 locations per
    stratum.
  * within a crop's assigned stratum, deterministic hash-based selection over
    eligible 12-voxel-separated interior centers (stable, reproducible).
  * exactly 3 edges/location (Z,Y,X), 36 total; 12 per axis by construction.
  * axis eligibility is INDEPENDENT of argmax(|gradient|).
  * exclude every exact affinity edge from history, including -006.
  * fresh -007 workspace/queue/event-log identities; nothing overwritten.

Records C_E_MAGNITUDE_CONTRAST_AVAILABLE_IN_FRESH_ELIGIBLE_POOL = false and the
observed contrast range, since fresh independent tissue lacks C/E-level contrast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SURVEY = REPO / "experiments/phase6e/MV-G3-NEW-REGION-SURVEY-001.json"
MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
PROTOCOL_OUT = REPO / "experiments/phase6e/G3_AXIS_NEUTRAL_SAMPLING_PROTOCOL_003.json"
WS_ROOT = REPO / "experiments/phase6e/g3-external-review-packages-007"
Q_ROOT = REPO / "experiments/phase6e/g3-interface-queues-007"

MARGIN = 3
SEP2 = 12 ** 2
STRATA = ["Q1", "Q2", "Q3", "Q4"]
PERCENTILE_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]

# Sources already used anywhere (existing regions + -005 + -006). Excluded to
# keep -007 independent. C/E and A-H are already excluded via the survey's
# excluded_existing_sources; -005/-006 added explicitly here.
USED_005 = {"MV-DVID-RAW-SURVEY-013", "MV-DVID-RAW-SURVEY-009", "MV-DVID-RAW-SURVEY-003", "MV-DVID-RAW-SURVEY-007"}
USED_006 = {"MV-DVID-RAW-SURVEY-010", "MV-DVID-RAW-SURVEY-014", "MV-DVID-RAW-SURVEY-019"}
N_CROPS = 12


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _hash_rank(*parts: str) -> int:
    """Deterministic stable hash for label-blind tie-breaking / selection."""
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16)


def historical_edges() -> set:
    """Every exact affinity edge (pair_left, pair_right, channel) ever queued or
    reviewed, including -006, to guarantee -007 independence."""
    import glob
    edges = set()
    for qp in glob.glob(str(REPO / "experiments/phase6e/g3-interface-queues-*/*.json")):
        try:
            data = json.loads(Path(qp).read_text())
        except Exception:
            continue
        for q in data.get("questions", []):
            edges.add((tuple(q["pair_left_zyx"]), tuple(q["pair_right_zyx"]), int(q["channel_zyx"])))
    for lp in glob.glob(str(REPO / "experiments/phase6e/g3-external-review-packages-*/*/workspace.events.jsonl")):
        for line in open(lp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if "pair_left_zyx" in e:
                edges.add((tuple(e["pair_left_zyx"]), tuple(e["pair_right_zyx"]), int(e["channel_zyx"])))
    return edges


def eligible_centers(arr: np.ndarray):
    """Interior centers where all three +1 edges are in-bounds, with local raw
    edge contrast = mean |c->c+1| over Z,Y,X. Returns (coords Nx3 int, contrast N).
    Order is a deterministic raster scan (C-order). Axis eligibility does NOT
    use argmax(|gradient|)."""
    z, y, x = arr.shape
    core = arr[MARGIN:z - MARGIN - 1, MARGIN:y - MARGIN - 1, MARGIN:x - MARGIN - 1]
    ez = np.abs(arr[MARGIN + 1:z - MARGIN, MARGIN:y - MARGIN - 1, MARGIN:x - MARGIN - 1] - core)
    ey = np.abs(arr[MARGIN:z - MARGIN - 1, MARGIN + 1:y - MARGIN, MARGIN:x - MARGIN - 1] - core)
    ex = np.abs(arr[MARGIN:z - MARGIN - 1, MARGIN:y - MARGIN - 1, MARGIN + 1:x - MARGIN] - core)
    contrast = (ez + ey + ex) / 3.0
    sz, sy, sx = contrast.shape
    zz, yy, xx = np.meshgrid(
        np.arange(MARGIN, MARGIN + sz), np.arange(MARGIN, MARGIN + sy), np.arange(MARGIN, MARGIN + sx), indexing="ij"
    )
    coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1).astype(np.int64)
    return coords, contrast.ravel().astype(np.float64)


def greedy_separated_centers(centers):
    """12-voxel spatially separated centers in raster-scan order. Accepts either
    the vectorized (coords, contrast) tuple or a list of (center, contrast).
    Separation predicate exact (d^2 < 144 reject). Returns list of (center, contrast)."""
    if isinstance(centers, tuple):
        coords, contrast = centers
        iterator = ((tuple(int(v) for v in coords[i]), float(contrast[i])) for i in range(len(contrast)))
    else:
        iterator = iter(centers)
    sep = 12
    grid = {}
    kept = []
    for (c, contrast_v) in iterator:
        z, y, x = c
        cell = (z // sep, y // sep, x // sep)
        conflict = False
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for (az, ay, ax) in grid.get((cell[0]+dz, cell[1]+dy, cell[2]+dx), ()):
                        if (az-z)**2 + (ay-y)**2 + (ax-x)**2 < SEP2:
                            conflict = True
                            break
                    if conflict: break
                if conflict: break
            if conflict: break
        if not conflict:
            grid.setdefault(cell, []).append((z, y, x))
            kept.append((c, contrast_v))
    return kept


def _edges(center):
    z, y, x = center
    return {
        "Z": [[z, y, x], [z + 1, y, x]],
        "Y": [[z, y, x], [z, y + 1, x]],
        "X": [[z, y, x], [z, y, x + 1]],
    }


def build():
    if PROTOCOL_OUT.exists() or WS_ROOT.exists() or Q_ROOT.exists():
        raise SystemExit("Refusing to overwrite existing -007 / protocol-003 artifacts")
    survey = json.loads(SURVEY.read_text())
    manifest = json.loads(MANIFEST.read_text())
    rec = {r["id"]: r for r in manifest["records"]}
    fresh = [s for s in survey["eligible_candidates"] if s not in USED_005 and s not in USED_006]
    fresh = sorted(fresh)  # canonical source-ID order, contrast-independent
    if len(fresh) < N_CROPS:
        raise SystemExit(f"Only {len(fresh)} fresh crops; need {N_CROPS}")
    chosen_sources = fresh[:N_CROPS]

    hist_edges = historical_edges()

    # 1) Freeze percentile boundaries per crop's own eligible contrast dist
    #    (relative within-eligible-population, computed BEFORE location pick).
    per_crop = []
    all_contrasts = []
    crop_centers = {}
    for sid in chosen_sources:
        arr = np.asarray(np.load(rec[sid]["raw_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        centers = greedy_separated_centers(eligible_centers(arr))
        crop_centers[sid] = centers
        all_contrasts.extend(c for _, c in centers)

    all_contrasts = np.array(all_contrasts)
    # population-level percentile boundaries (relative, frozen now)
    bounds = [float(np.quantile(all_contrasts, p)) for p in PERCENTILE_EDGES]

    def stratum_of(contrast):
        for i in range(4):
            lo, hi = bounds[i], bounds[i + 1]
            if (contrast >= lo and contrast < hi) or (i == 3 and contrast <= hi):
                return STRATA[i]
        return STRATA[-1]

    # 2) Fixed stratum rotation by crop order
    assignments = [STRATA[i % 4] for i in range(N_CROPS)]

    locations = []
    for i, sid in enumerate(chosen_sources):
        target_stratum = assignments[i]
        candidates = [(c, contrast) for (c, contrast) in crop_centers[sid] if stratum_of(contrast) == target_stratum]
        # exclude any center whose Z/Y/X edges collide with history
        def ok(center):
            for ax_i, ax in enumerate(["Z", "Y", "X"]):
                le, ri = _edges(center)[ax]
                if (tuple(le), tuple(ri), ax_i) in hist_edges:
                    return False
            return True
        candidates = [(c, ct) for (c, ct) in candidates if ok(c)]
        if not candidates:
            raise SystemExit(f"{sid}: no eligible center in stratum {target_stratum} after history exclusion")
        # deterministic hash-based pick using immutable raw sha + source id + coords
        raw_sha = rec[sid]["raw_sha256"]
        pick = min(candidates, key=lambda cc: _hash_rank(raw_sha, sid, target_stratum, str(cc[0])))
        center, contrast = pick
        # percentile rank of the chosen contrast within population
        pctile = float((all_contrasts < contrast).mean())
        locations.append({
            "index": i + 1,
            "crop_id": f"MV-G3-AXNEU3-{i + 1:02d}",
            "source_id": sid,
            "raw_sha256": raw_sha,
            "assigned_stratum": target_stratum,
            "center_zyx": list(center),
            "raw_contrast": contrast,
            "contrast_percentile_rank": pctile,
            "edges": _edges(center),
        })

    # protocol doc
    protocol = {
        "id": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-003",
        "schema_version": 1,
        "created_at": _now(),
        "status": "PREDECLARED_FIXED_BEFORE_LABELS",
        "primary_question": "At the same fresh physical location, do independently reviewed Z/Y/X adjacent affinity edges retain the historical axis/class relationship?",
        "sampler_id": "AXIS_NEUTRAL_PAIRED_LOCATION_V3",
        "n_crops": N_CROPS,
        "locations_per_crop": 1,
        "edges_per_location": 3,
        "total_edges": 3 * N_CROPS,
        "designed_axis_balance": {"Z": N_CROPS, "Y": N_CROPS, "X": N_CROPS},
        "contrast_strata": {
            "definition": "relative within-eligible-population percentile ranks of raw-only local edge contrast (mean |c->c+1| over Z,Y,X)",
            "labels": STRATA,
            "percentile_edges": PERCENTILE_EDGES,
            "population_contrast_boundaries": bounds,
            "boundaries_frozen_before_location_selection": True,
        },
        "crop_ordering": "canonical survey source-ID ascending (contrast-independent)",
        "stratum_rotation": assignments,
        "location_selection": "within assigned stratum, deterministic sha256 hash-rank over 12-voxel-separated interior centers using immutable raw_sha256+source_id+stratum+coords",
        "edge_eligibility": "all three edges eligible independently; NOT gated by argmax(|gradient|)",
        "excluded_sources": {"existing_regions_A_H_C_E_and_validation": survey["excluded_existing_sources"], "used_005": sorted(USED_005), "used_006": sorted(USED_006)},
        "historical_edges_excluded_count": len(hist_edges),
        "C_E_MAGNITUDE_CONTRAST_AVAILABLE_IN_FRESH_ELIGIBLE_POOL": False,
        "observed_fresh_contrast_range": {"min": float(all_contrasts.min()), "p50": float(np.median(all_contrasts)), "max": float(all_contrasts.max()), "boundaries": bounds},
        "scope_limitation": "Tests axis dependence across the contrast distribution available in FRESH INDEPENDENT tissue. Does NOT establish behavior in the historical C/E absolute-contrast regime.",
        "forbidden_inputs": ["SAME/DIFFERENT labels", "class predictions", "learned affinities", "historical class composition", "argmax-gradient edge eligibility", "C/E or any used/reviewed crop"],
        "locations": locations,
    }
    PROTOCOL_OUT.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # workspaces + queues, presentation interleaved by (location, axis)
    WS_ROOT.mkdir(parents=True)
    Q_ROOT.mkdir(parents=True)
    summary = []
    for loc in locations:
        crop_id = loc["crop_id"]
        sid = loc["source_id"]
        r = rec[sid]
        crop_dir = WS_ROOT / crop_id
        crop_dir.mkdir(parents=True)
        log = crop_dir / "workspace.events.jsonl"
        log.write_text("", encoding="utf-8")
        ws_id = f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id}-007"
        workspace = {
            "allowed_decisions": ["BAD_QUESTION", "DIFFERENT_PROCESS", "SAME_PROCESS", "UNCERTAIN"],
            "coordinate_frame": "MV-FRAME-DVID-WASP5-001",
            "created_at": _now(),
            "crop_id": crop_id,
            "event_log": {"append_only": True, "path": str(log.resolve())},
            "id": ws_id,
            "pair_contract": "pair_left_zyx is followed by its positive neighbour along channel_zyx: 0=Z, 1=Y, 2=X.",
            "parent_region_id": sid,
            "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
            "raw": {"path": r["raw_path"], "sha256": r["raw_sha256"], "shape_zyx": r["shape_zyx"], "dtype": r["dtype"]},
            "provenance": {"protocol": {"path": str(PROTOCOL_OUT.resolve()), "id": protocol["id"]}, "sampler": protocol["sampler_id"], "assigned_stratum": loc["assigned_stratum"], "center_zyx": loc["center_zyx"]},
            "review_state": "UNREVIEWED",
            "schema_version": 1,
            "source_origin_xyz": [r["bounds_xyz"][a][0] for a in ("x", "y", "z")],
            "split": "G3_TARGET_TRAIN",
            "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
        }
        (crop_dir / "workspace.json").write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # one interface, three members = the three axis edges at this location
        axis_index = {"Z": 0, "Y": 1, "X": 2}
        questions = []
        for m, ax in enumerate(["Z", "Y", "X"], 1):
            le, ri = loc["edges"][ax]
            questions.append({
                "id": f"MV-G3-AXNEU3-{loc['index']:02d}-{ax}", "kind": "RAW_EM_AXIS_NEUTRAL_EDGE",
                "interface_id": f"MV-G3-AXNEU3-{loc['index']:02d}", "interface_member": m, "interface_members": 3,
                "pair_left_zyx": le, "pair_right_zyx": ri, "channel_zyx": axis_index[ax], "axis_name": ax,
                "assigned_stratum": loc["assigned_stratum"], "raw_contrast": loc["raw_contrast"],
                "contrast_percentile_rank": loc["contrast_percentile_rank"],
                "raw_gradient_score": loc["raw_contrast"],  # reviewer displays this field
                "selection": "AXIS_NEUTRAL_PAIRED_LOCATION_V3", "model_navigation": False,
            })
        queue = {
            "schema_version": 1, "id": f"MV-G3-AXNEU3-QUEUE-{crop_id}-007", "created_at": _now(),
            "status": "EXPERT_INTERFACE_REVIEW_REQUIRED", "workspace_id": ws_id, "crop_id": crop_id,
            "raw_sha256": r["raw_sha256"],
            "selection": {"method": "AXIS_NEUTRAL_PAIRED_LOCATION_V3", "protocol": protocol["id"],
                          "assigned_stratum": loc["assigned_stratum"], "center_zyx": loc["center_zyx"],
                          "label_blind": True, "argmax_gradient_used": False},
            "questions": questions,
            "scientific_boundary": "Paired Z/Y/X edges at one physical location; each independently reviewed. Location chosen by raw-only contrast stratum + deterministic hash; no class targeted.",
        }
        (Q_ROOT / f"{crop_id}.json").write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
        summary.append({"crop_id": crop_id, "source_id": sid, "stratum": loc["assigned_stratum"], "center_zyx": loc["center_zyx"], "workspace_id": ws_id})

    batch = {"id": "MV-G3-AXIS-NEUTRAL-REVIEW-BATCH-007", "created_at": _now(), "protocol": protocol["id"],
             "locations": len(locations), "total_edges": 3 * len(locations), "per_crop": summary,
             "C_E_MAGNITUDE_CONTRAST_AVAILABLE_IN_FRESH_ELIGIBLE_POOL": False}
    (Q_ROOT / "batch-manifest.json").write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"locations": len(locations), "total_edges": 3 * len(locations),
                      "strata": {s: sum(1 for l in locations if l["assigned_stratum"] == s) for s in STRATA},
                      "crops": [l["source_id"] for l in locations],
                      "C_E_contrast_available": False,
                      "observed_contrast_range": [float(all_contrasts.min()), float(all_contrasts.max())]}, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
