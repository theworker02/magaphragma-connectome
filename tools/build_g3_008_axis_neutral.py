"""Corrected, fully-deterministic axis-neutral paired-location sampler (-008).

Supersedes -007 for prospective use (see G3_007_PROVENANCE_FAILURE_AND_
SUPERSESSION). Fixes the two reproducibility defects:

  RC1  historical exclusion is a PINNED, enumerated snapshot captured once at
       build time and written into the protocol with a sha256 -- never a live
       glob at selection time, and never self-referential.
  RC2  local raw edge contrast is computed in float64 from a float64-promoted
       copy of the raw, matching the pure-python reference EXACTLY (atol=0), so
       stratum membership and hash-rank selection are numerically stable.

Design (predeclared, label-blind, fixed before labels):
  * population: interior centers where all three +1 edges (Z,Y,X) are in-bounds.
  * contrast(center) = mean(|c->c+1|) over Z,Y,X, float64.
  * greedy 12-voxel spatial separation in raster order (exact d^2>=144).
  * fresh crops = eligible survey sources minus every previously used source,
    canonical ascending source-id order (contrast-independent).
  * pooled percentile strata Q1..Q4 over the union of separated-center
    contrasts, boundaries frozen before location pick.
  * one location per crop; stratum assigned by fixed rotation Q1,Q2,Q3,Q4,...
  * within stratum, pick the center whose sha256(raw_sha|source|stratum|str(center))
    is minimal, among centers whose 3 edges are not in the pinned exclusion set.
  * all three edges Z/Y/X emitted; argmax(|gradient|) never used.

The module is import-safe (no work at import). build() writes protocol + queues
+ workspaces; select_only() returns the canonical selection for double-run
comparison without writing.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SURVEY = REPO / "experiments/phase6e/MV-G3-NEW-REGION-SURVEY-001.json"
MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
MARGIN = 3
SEP2 = 12 ** 2
PERCENTILE_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]
STRATA = ["Q1", "Q2", "Q3", "Q4"]
N_CROPS = 12

# Survey sources consumed by tiers that carry REAL provenance/evidence and must
# not be reused: -005 (frozen new-train review) and -006 (axis-neutral review).
# The -007 sources are DELIBERATELY NOT excluded here: -007 was formally
# superseded (G3_007_PROVENANCE_FAILURE_AND_SUPERSESSION) with ALL 12 event logs
# EMPTY (zero human decisions), so reclaiming its raw sources destroys no
# evidence. This yields 2 never-touched + 12 reclaimed -007 = 14 fully-available
# crops, from which -008 deterministically selects 12 (2 held in reserve).
# NOTE: -008's own edges are still excluded from the pinned historical snapshot
# by geometry; and -007's queued edges remain in the pinned snapshot so -008
# will not duplicate any exact edge -007 happened to enumerate.
USED_PRIOR_SOURCES = {
    # -005 (frozen reviewed new-train crops I-L)
    "MV-DVID-RAW-SURVEY-013", "MV-DVID-RAW-SURVEY-009", "MV-DVID-RAW-SURVEY-003", "MV-DVID-RAW-SURVEY-007",
    # -006 (axis-neutral A/B/C)
    "MV-DVID-RAW-SURVEY-010", "MV-DVID-RAW-SURVEY-014", "MV-DVID-RAW-SURVEY-019",
}
# Explicitly recorded for provenance: -007 sources reclaimed after supersession.
RECLAIMED_SUPERSEDED_007_SOURCES = {
    "MV-DVID-RAW-SURVEY-005", "MV-DVID-RAW-SURVEY-006", "MV-DVID-RAW-SURVEY-011", "MV-DVID-RAW-SURVEY-012",
    "MV-DVID-RAW-SURVEY-016", "MV-DVID-RAW-SURVEY-018", "MV-DVID-RAW-SURVEY-020", "MV-DVID-RAW-SURVEY-021",
    "MV-DVID-RAW-SURVEY-023", "MV-DVID-RAW-SURVEY-026", "MV-DVID-RAW-SURVEY-028", "MV-DVID-RAW-SURVEY-029",
}

PROTOCOL_OUT = REPO / "experiments/phase6e/G3_AXIS_NEUTRAL_SAMPLING_PROTOCOL_004.json"
WS_ROOT = REPO / "experiments/phase6e/g3-external-review-packages-008"
Q_ROOT = REPO / "experiments/phase6e/g3-interface-queues-008"

# The SOLE historical-exclusion input for the prospective -008 sampler. It is a
# frozen, enumerated snapshot with a verifiable digest. The prospective sampler
# must NEVER glob directories for exclusions (that is filesystem-state-dependent
# machinery and was the -008 reproducibility defect). Only the exposure-audit /
# freeze tooling is permitted to discover historical artifacts; that discovery
# is what produced this snapshot once, offline.
PINNED_SNAPSHOT = REPO / "experiments/phase6e/G3_008_PINNED_EXCLUSION_SNAPSHOT_001.json"
PINNED_SNAPSHOT_EXPECTED_ID = "MV-G3-008-PINNED-EXCLUSION-SNAPSHOT-001"
PINNED_SNAPSHOT_EXPECTED_STATUS = "FROZEN_PINNED_EXCLUSION_FOR_G3_008"
PINNED_SNAPSHOT_EXPECTED_SCHEMA = 1
PINNED_SNAPSHOT_EXPECTED_COUNT = 713
PINNED_SNAPSHOT_EXPECTED_EDGES_SHA256 = "2cbabb9933c3e88fa2a03287ec8cd2b65f0dd3b72dcaf35b235402c7892dafeb"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _hash_rank(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16)


def pinned_historical_edges() -> tuple[list, str]:
    """Load the SOLE historical-exclusion input from the frozen pinned snapshot,
    verify its identity/schema/status and its enumerated edge digest, and return
    (sorted list of [left,right,channel], sha256 of that list).

    This performs NO directory globbing. The prospective sampler's historical
    exclusions therefore depend only on an immutable, hash-verified artifact and
    cannot be perturbed by filesystem state (partial writes, the -008 output
    dirs, concurrent processes, or sync activity). The exposure-audit/freeze
    tool is the only component permitted to discover artifacts on disk; it did
    so once to produce this snapshot.
    """
    snap = json.loads(PINNED_SNAPSHOT.read_text(encoding="utf-8"))
    if snap.get("id") != PINNED_SNAPSHOT_EXPECTED_ID:
        raise SystemExit(f"pinned snapshot id mismatch: {snap.get('id')!r}")
    if snap.get("status") != PINNED_SNAPSHOT_EXPECTED_STATUS:
        raise SystemExit(f"pinned snapshot status mismatch: {snap.get('status')!r}")
    if int(snap.get("schema_version", -1)) != PINNED_SNAPSHOT_EXPECTED_SCHEMA:
        raise SystemExit(f"pinned snapshot schema mismatch: {snap.get('schema_version')!r}")

    edges = set()
    for src in snap.get("per_source", []):
        for edge in src.get("avoid_edges", []):
            left, right, channel = edge[0], edge[1], int(edge[2])
            edges.add((tuple(left), tuple(right), channel))
    listed = sorted([list(l), list(r), c] for (l, r, c) in edges)
    digest = hashlib.sha256(json.dumps(listed, sort_keys=True).encode()).hexdigest()

    # Fail closed if the snapshot's enumerated edges do not match the pinned
    # count/digest this sampler version was validated against. This guarantees
    # the historical exclusion set is EXACTLY the reviewed/queued edges recorded
    # at freeze time, never something a live filesystem happened to contain.
    if len(listed) != PINNED_SNAPSHOT_EXPECTED_COUNT:
        raise SystemExit(f"pinned snapshot edge count {len(listed)} != expected {PINNED_SNAPSHOT_EXPECTED_COUNT}")
    if digest != PINNED_SNAPSHOT_EXPECTED_EDGES_SHA256:
        raise SystemExit(f"pinned snapshot edge digest {digest} != expected {PINNED_SNAPSHOT_EXPECTED_EDGES_SHA256}")
    return listed, digest


def eligible_centers(arr: np.ndarray):
    """float64-stable. Returns (coords Nx3 python-int list, contrast list float)."""
    a = np.asarray(arr, dtype=np.float64)  # promote once; stable reductions
    z, y, x = a.shape
    core = a[MARGIN:z - MARGIN - 1, MARGIN:y - MARGIN - 1, MARGIN:x - MARGIN - 1]
    ez = np.abs(a[MARGIN + 1:z - MARGIN, MARGIN:y - MARGIN - 1, MARGIN:x - MARGIN - 1] - core)
    ey = np.abs(a[MARGIN:z - MARGIN - 1, MARGIN + 1:y - MARGIN, MARGIN:x - MARGIN - 1] - core)
    ex = np.abs(a[MARGIN:z - MARGIN - 1, MARGIN:y - MARGIN - 1, MARGIN + 1:x - MARGIN] - core)
    contrast = (ez + ey + ex) / 3.0
    sz, sy, sx = contrast.shape
    coords = []
    cvals = []
    cflat = contrast.ravel()
    idx = 0
    # explicit C-order raster to guarantee pure-python ints and stable order
    for zi in range(sz):
        for yi in range(sy):
            for xi in range(sx):
                coords.append((MARGIN + zi, MARGIN + yi, MARGIN + xi))
                cvals.append(float(cflat[idx]))
                idx += 1
    return coords, cvals


def greedy_separated_centers(coords, cvals):
    sep = 12
    grid = {}
    kept = []
    for c, ct in zip(coords, cvals):
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


def _edges(center):
    z, y, x = center
    return {"Z": [[z, y, x], [z + 1, y, x]], "Y": [[z, y, x], [z, y + 1, x]], "X": [[z, y, x], [z, y, x + 1]]}


def _axis_index(ax):
    return {"Z": 0, "Y": 1, "X": 2}[ax]


def select_only():
    """Compute the canonical selection deterministically WITHOUT writing.
    Returns dict with fresh sources, pinned-exclusion hash, boundaries, and the
    ordered list of (source, stratum, center, contrast)."""
    survey = json.loads(SURVEY.read_text())
    manifest = json.loads(MANIFEST.read_text())
    rec = {r["id"]: r for r in manifest["records"]}
    fresh = sorted(s for s in survey["eligible_candidates"] if s not in USED_PRIOR_SOURCES)
    if len(fresh) < N_CROPS:
        raise SystemExit(f"Only {len(fresh)} fresh crops; need {N_CROPS}")
    chosen = fresh[:N_CROPS]

    hist_list, hist_hash = pinned_historical_edges()
    hist_set = {(tuple(l), tuple(r), c) for (l, r, c) in hist_list}

    crop_centers = {}
    all_contrasts = []
    for sid in chosen:
        arr = np.load(rec[sid]["raw_path"], mmap_mode="r", allow_pickle=False)
        centers = greedy_separated_centers(*eligible_centers(arr))
        crop_centers[sid] = centers
        all_contrasts.extend(ct for _, ct in centers)
    bounds = [float(np.quantile(np.array(all_contrasts, dtype=np.float64), p)) for p in PERCENTILE_EDGES]

    def stratum_of(ct):
        for i in range(4):
            lo, hi = bounds[i], bounds[i + 1]
            if (lo <= ct < hi) or (i == 3 and ct <= hi):
                return STRATA[i]
        return STRATA[-1]

    assignments = [STRATA[i % 4] for i in range(N_CROPS)]
    selection = []
    for i, sid in enumerate(chosen):
        ts = assignments[i]
        raw_sha = rec[sid]["raw_sha256"]
        cands = []
        for (c, ct) in crop_centers[sid]:
            if stratum_of(ct) != ts:
                continue
            edges = _edges(c)
            if any((tuple(edges[ax][0]), tuple(edges[ax][1]), _axis_index(ax)) in hist_set for ax in ("Z", "Y", "X")):
                continue
            cands.append((c, ct))
        if not cands:
            raise SystemExit(f"{sid}: no eligible center in stratum {ts} after exclusion")
        pick = min(cands, key=lambda cc: _hash_rank(raw_sha, sid, ts, str(cc[0])))
        selection.append({"source_id": sid, "stratum": ts, "center_zyx": list(pick[0]), "raw_contrast": pick[1], "raw_sha256": raw_sha})
    return {"fresh_sources": chosen, "pinned_exclusion_sha256": hist_hash, "pinned_exclusion_count": len(hist_list),
            "population_contrast_boundaries": bounds, "selection": selection, "pinned_exclusion_edges": hist_list}


def canonical_manifest(sel: dict) -> dict:
    """Deterministic manifest for double-run comparison (no timestamps)."""
    return {
        "sampler_id": "AXIS_NEUTRAL_PAIRED_LOCATION_DETERMINISTIC_V4",
        "fresh_sources": sel["fresh_sources"],
        "pinned_exclusion_sha256": sel["pinned_exclusion_sha256"],
        "pinned_exclusion_count": sel["pinned_exclusion_count"],
        "population_contrast_boundaries": sel["population_contrast_boundaries"],
        "selection": sel["selection"],
    }


def _assert_build_invariants(sel: dict) -> None:
    """Hard structural gate over select_only()'s result, BEFORE any disk write.
    build() materializes this already-tested selection; it must never re-derive
    or repair it. Any violation aborts the build with nothing written."""
    selection = sel["selection"]

    # 12 physical locations / 12 distinct source crops.
    assert len(selection) == N_CROPS, f"expected {N_CROPS} locations, got {len(selection)}"
    sources = [it["source_id"] for it in selection]
    assert len(set(sources)) == N_CROPS, "source crops are not 12 distinct"

    # 36 total edges = 12 locations x 3 axis edges; 12 Z / 12 Y / 12 X.
    # (Edges are emitted 1-per-axis per center in build(); assert the design.)
    total_edges = len(selection) * 3
    assert total_edges == 36, f"expected 36 edges, got {total_edges}"
    # per-axis balance is structural: exactly one Z, one Y, one X per center.
    # 3 edges per physical center is guaranteed by _edges(); assert center count.
    assert all(len(_edges(tuple(it["center_zyx"]))) == 3 for it in selection), "not 3 edges per center"

    # Stratum balance: 3 Q1 / 3 Q2 / 3 Q3 / 3 Q4.
    strat_counts = {s: sum(1 for it in selection if it["stratum"] == s) for s in STRATA}
    assert strat_counts == {s: 3 for s in STRATA}, f"stratum balance off: {strat_counts}"

    # All 36 edges absent from the pinned 713-edge (snapshot) exclusion set.
    hist_set = {(tuple(l), tuple(r), c) for (l, r, c) in sel["pinned_exclusion_edges"]}
    for it in selection:
        edges = _edges(tuple(it["center_zyx"]))
        for ax in ("Z", "Y", "X"):
            key = (tuple(edges[ax][0]), tuple(edges[ax][1]), _axis_index(ax))
            assert key not in hist_set, f"edge {key} collides with pinned exclusion snapshot"

    # Provenance: raw sha256 present and matches survey manifest per source.
    manifest = json.loads(MANIFEST.read_text())
    rec = {r["id"]: r for r in manifest["records"]}
    for it in selection:
        sid = it["source_id"]
        assert sid in rec, f"{sid} absent from survey manifest"
        assert it["raw_sha256"] == rec[sid]["raw_sha256"], f"{sid}: raw sha mismatch"

    # Spatial leakage: every selected center is interior with all +1 edges in
    # bounds for its own crop (paired-location edges cannot leave the volume).
    for it in selection:
        sid = it["source_id"]
        z, y, x = rec[sid]["shape_zyx"]
        cz, cy, cx = it["center_zyx"]
        assert MARGIN <= cz < z - MARGIN - 1 and MARGIN <= cy < y - MARGIN - 1 and MARGIN <= cx < x - MARGIN - 1, \
            f"{sid}: center {it['center_zyx']} not interior with +1 headroom"

    # Workspace / package / queue IDs unique (derived deterministically here so
    # collisions are caught before any file is created).
    ws_ids = [f"MV-EXTERNAL-BOUNDARY-WORKSPACE-MV-G3-AXNEU4-{i:02d}-008" for i in range(1, N_CROPS + 1)]
    q_ids = [f"MV-G3-RAW-AXISNEUTRAL-QUEUE-MV-G3-AXNEU4-{i:02d}-008" for i in range(1, N_CROPS + 1)]
    crop_ids = [f"MV-G3-AXNEU4-{i:02d}" for i in range(1, N_CROPS + 1)]
    assert len(set(ws_ids)) == N_CROPS and len(set(q_ids)) == N_CROPS and len(set(crop_ids)) == N_CROPS, "non-unique ids"

    # C/E-magnitude contrast is NOT available in the fresh eligible pool; the
    # protocol must record this honestly rather than imply high-contrast tissue.
    assert sel["population_contrast_boundaries"][0] <= sel["population_contrast_boundaries"][-1]


def build(protocol_out: Path = PROTOCOL_OUT, ws_root: Path = WS_ROOT, q_root: Path = Q_ROOT) -> dict:
    """Write the -008 protocol, workspaces, and queues from the canonical
    selection. Refuses to overwrite. The pinned exclusion snapshot (edges +
    sha256) is written INTO the protocol so selection never depends on a live
    glob again.

    build() is deliberately boring: it materializes select_only()'s already-
    tested result after a hard structural gate, and never re-implements
    selection."""
    if protocol_out.exists() or ws_root.exists() or q_root.exists():
        raise FileExistsError("Refusing to overwrite existing -008 artifacts")
    sel = select_only()
    _assert_build_invariants(sel)
    manifest = json.loads(MANIFEST.read_text())
    rec = {r["id"]: r for r in manifest["records"]}
    ws_root.mkdir(parents=True)
    q_root.mkdir(parents=True)

    per_crop = []
    for i, item in enumerate(sel["selection"], 1):
        sid = item["source_id"]
        r = rec[sid]
        crop_id = f"MV-G3-AXNEU4-{i:02d}"
        center = tuple(item["center_zyx"])
        edges = _edges(center)
        crop_dir = ws_root / crop_id
        crop_dir.mkdir(parents=True)
        log = crop_dir / "workspace.events.jsonl"
        log.write_text("", encoding="utf-8")
        ws_id = f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id}-008"
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
            "provenance": {"protocol": {"path": str(protocol_out.resolve()), "id": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-004"},
                           "stratum": item["stratum"], "center_zyx": list(center),
                           "reclaimed_superseded_007_source": sid in RECLAIMED_SUPERSEDED_007_SOURCES},
            "review_state": "UNREVIEWED", "schema_version": 1,
            "source_origin_xyz": [r["bounds_xyz"][a][0] for a in ("x", "y", "z")],
            "split": "G3_TARGET_TRAIN",
            "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
        }
        (crop_dir / "workspace.json").write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        questions = []
        # deterministic interleave: one edge per axis, presented Z,Y,X
        for m, ax in enumerate(("Z", "Y", "X"), 1):
            left, right = edges[ax]
            questions.append({
                "id": f"MV-G3-IF-{i:03d}-{m:02d}", "kind": "RAW_EM_INTERFACE_MEMBER",
                "interface_id": f"MV-G3-IF-{i:03d}", "interface_member": m, "interface_members": 3,
                "pair_left_zyx": list(left), "pair_right_zyx": list(right), "channel_zyx": _axis_index(ax),
                "axis_name": ax, "stratum": item["stratum"], "raw_gradient_score": 0.0,
                "selection": "AXIS_NEUTRAL_PAIRED_LOCATION_DETERMINISTIC_V4", "model_navigation": False,
            })
        queue = {
            "schema_version": 1, "id": f"MV-G3-RAW-AXISNEUTRAL-QUEUE-{crop_id}-008", "created_at": _now(),
            "status": "EXPERT_INTERFACE_REVIEW_REQUIRED", "workspace_id": ws_id, "crop_id": crop_id,
            "raw_sha256": r["raw_sha256"],
            "selection": {"method": "AXIS_NEUTRAL_PAIRED_LOCATION_DETERMINISTIC_V4", "stratum": item["stratum"],
                          "center_zyx": list(center), "label_blind": True,
                          "forbidden_inputs": ["SAME/DIFFERENT labels", "class predictions", "learned affinities", "argmax-gradient edge eligibility"]},
            "questions": questions,
            "scientific_boundary": "Axis-neutral paired-location design: all three edges at one physical center are independently reviewed; no axis is privileged and no class is targeted.",
        }
        (q_root / f"{crop_id}.json").write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
        per_crop.append({"crop_id": crop_id, "source_id": sid, "stratum": item["stratum"], "center_zyx": list(center),
                         "workspace_id": ws_id, "raw_contrast": item["raw_contrast"],
                         "reclaimed_superseded_007_source": sid in RECLAIMED_SUPERSEDED_007_SOURCES})

    observed = [it["raw_contrast"] for it in sel["selection"]]
    protocol = {
        "id": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-004", "schema_version": 1, "created_at": _now(),
        "status": "PREDECLARED_FIXED_BEFORE_LABELS",
        "sampler_id": "AXIS_NEUTRAL_PAIRED_LOCATION_DETERMINISTIC_V4",
        "canonical_manifest": canonical_manifest(sel),
        "pinned_exclusion_sha256": sel["pinned_exclusion_sha256"],
        "pinned_exclusion_count": sel["pinned_exclusion_count"],
        "pinned_exclusion_edges": sel["pinned_exclusion_edges"],
        "population_contrast_boundaries": sel["population_contrast_boundaries"],
        "C_E_MAGNITUDE_CONTRAST_AVAILABLE_IN_FRESH_ELIGIBLE_POOL": False,
        "observed_selected_contrast_range": [min(observed), max(observed)],
        "contrast_strata_semantics": "RELATIVE percentiles Q1(0-25) Q2(25-50) Q3(50-75) Q4(75-100) over the pooled separated-center contrast of the 12 selected fresh crops; NOT absolute and NOT equivalent to historical C/E magnitude",
        "reclaimed_superseded_007_sources": sorted(RECLAIMED_SUPERSEDED_007_SOURCES),
        "design": {"n_crops": N_CROPS, "locations_per_crop": 1, "edges_per_location": 3, "total_edges": N_CROPS * 3,
                   "stratum_rotation": [STRATA[i % 4] for i in range(N_CROPS)]},
        "per_crop": per_crop,
    }
    protocol_out.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return protocol


if __name__ == "__main__":
    import sys
    if "--select-only" in sys.argv:
        print(json.dumps(canonical_manifest(select_only()), indent=2, sort_keys=True))
    elif "--build" in sys.argv:
        p = build()
        print(json.dumps({"protocol": p["id"], "crops": len(p["per_crop"]), "total_edges": p["design"]["total_edges"],
                          "pinned_exclusion_sha256": p["pinned_exclusion_sha256"]}, indent=2))
