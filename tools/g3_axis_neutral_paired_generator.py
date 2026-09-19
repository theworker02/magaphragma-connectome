"""Axis-neutral, paired-location interface generator (H3 diagnostic).

NEW generator version; does NOT modify the historical generator. It removes
`axis = argmax(abs(gradient))` as the mechanism deciding which affinity edge is
eligible. At each selected center location it defines ALL THREE adjacent edges
independently:
    Z: (z,y,x) <-> (z+1,y,x)
    Y: (z,y,x) <-> (z,y+1,x)
    X: (z,y,x) <-> (z,y,x+1)
Edge existence/eligibility does NOT depend on which axis has the strongest
gradient. Raw gradient/intensity is used ONLY for predeclared label-blind
stratification of LOCATIONS into raw edge-contrast bands.

Paired design: at one center the reviewer independently labels the Z, Y, and X
edges from the SAME local neighborhood (controls crop identity + local
morphology). The three axes are NOT merged into one decision. Presentation
order is deterministically interleaved so the reviewer is not shown all-Z then
all-Y then all-X. Queue metadata retains the true axis for provenance.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_OUT = REPO / "experiments/phase6e/G3_AXIS_NEUTRAL_SAMPLING_PROTOCOL_001.json"
MARGIN = 3
SEP = 12
SEP2 = SEP * SEP
# Predeclared raw edge-contrast bands (mean |1-voxel step| over the location's
# three axis edges), fixed BEFORE labels. Band edges in raw intensity units.
CONTRAST_BANDS = [
    {"label": "very_low", "lo": 0.0, "hi": 5.0},
    {"label": "low", "lo": 5.0, "hi": 15.0},
    {"label": "mid", "lo": 15.0, "hi": 40.0},
    {"label": "high", "lo": 40.0, "hi": 1e9},
]
LOCATIONS_PER_CROP = 3  # 3 crops x 3 locations x 3 axes = 27 edge decisions (in 24-36 target)

DECLARED_RULE = {
    "sampler_id": "AXIS_NEUTRAL_PAIRED_LOCATION_V1",
    "declared_before_labels": True,
    "edge_eligibility": "at each selected interior center, all three edges (Z,Y,X) are eligible INDEPENDENTLY; eligibility does NOT depend on argmax(|gradient|)",
    "edges": {"Z": "(z,y,x)<->(z+1,y,x)", "Y": "(z,y,x)<->(z,y+1,x)", "X": "(z,y,x)<->(z,y,x+1)"},
    "location_selection": "interior centers, 12-voxel spatially separated, deterministically drawn to spread across predeclared raw edge-contrast bands",
    "location_stratifier": "mean absolute single-voxel intensity step across the location's three axis edges (raw EM only)",
    "contrast_bands": CONTRAST_BANDS,
    "locations_per_crop": LOCATIONS_PER_CROP,
    "presentation_order": "deterministic interleave of (location, axis) so axes are not shown in blocks",
    "allowed_inputs": ["raw EM intensity", "raw gradient for stratification only", "crop identity", "geometry"],
    "forbidden_inputs": ["argmax-gradient edge eligibility", "SAME/DIFFERENT labels", "class predictions", "learned affinities", "historical class composition"],
    "purpose": "test H3 (dominant-gradient selection) vs H4 (intrinsic Z-edge effect) with balanced Z/Y/X edge eligibility at matched locations",
}


def _now():
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _edges(centre):
    z, y, x = centre
    return {
        0: ((z, y, x), (z + 1, y, x)),
        1: ((z, y, x), (z, y + 1, x)),
        2: ((z, y, x), (z, y, x + 1)),
    }


def location_contrast(arr, centre):
    """Mean |1-voxel step| across the three axis edges at this center (raw EM)."""
    steps = []
    for axis, (a, b) in _edges(centre).items():
        steps.append(abs(float(arr[a]) - float(arr[b])))
    return float(np.mean(steps))


def eligible_centers(arr):
    """All interior centers whose 3 edges are fully in-bounds. Returns coords."""
    z0, y0, x0 = MARGIN, MARGIN, MARGIN
    z1, y1, x1 = arr.shape[0] - MARGIN - 1, arr.shape[1] - MARGIN - 1, arr.shape[2] - MARGIN - 1
    zs, ys, xs = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    coords = np.stack([zs.ravel(), ys.ravel(), xs.ravel()], axis=1)
    return coords


def select_locations(arr, prior_edges, per_crop):
    """Deterministic label-blind selection: for each contrast band, pick the
    highest-contrast interior separated center in that band, until per_crop
    locations chosen with band coverage. Excludes any center whose ANY edge is
    a previously reviewed exact edge."""
    coords = eligible_centers(arr)
    # compute location contrast for a bounded candidate set: rank all centers by
    # contrast descending is expensive; instead sample deterministically by
    # scanning in raster order but bucketed by band. Compute contrast lazily.
    # For determinism + tractability, evaluate contrast on all centers.
    contrasts = np.array([location_contrast(arr, tuple(int(v) for v in c)) for c in coords])
    order = np.argsort(contrasts, kind="stable")[::-1]

    chosen = []
    chosen_centres = []
    # round-robin over bands: aim for even band coverage
    band_targets = {b["label"]: 0 for b in CONTRAST_BANDS}
    # distribute per_crop across bands as evenly as possible
    for i in range(per_crop):
        band_targets[CONTRAST_BANDS[i % len(CONTRAST_BANDS)]["label"]] += 1

    def band_of(v):
        for b in CONTRAST_BANDS:
            if b["lo"] <= v < b["hi"]:
                return b["label"]
        return CONTRAST_BANDS[-1]["label"]

    picked_per_band = {b["label"]: 0 for b in CONTRAST_BANDS}
    for idx in order:
        if len(chosen) >= per_crop:
            break
        centre = tuple(int(v) for v in coords[idx])
        v = float(contrasts[idx])
        band = band_of(v)
        if picked_per_band[band] >= band_targets[band]:
            continue
        # exclude if any of its three edges was previously reviewed
        edges = _edges(centre)
        if any((e[0], e[1], axis) in prior_edges for axis, e in edges.items()):
            continue
        # separation vs chosen this crop
        if any((centre[0]-c[0])**2 + (centre[1]-c[1])**2 + (centre[2]-c[2])**2 < SEP2 for c in chosen_centres):
            continue
        chosen.append({"centre": centre, "contrast": v, "band": band})
        chosen_centres.append(centre)
        picked_per_band[band] += 1

    # if some bands under-filled (sparse), backfill from remaining high-contrast
    if len(chosen) < per_crop:
        for idx in order:
            if len(chosen) >= per_crop:
                break
            centre = tuple(int(v) for v in coords[idx])
            if centre in chosen_centres:
                continue
            edges = _edges(centre)
            if any((e[0], e[1], axis) in prior_edges for axis, e in edges.items()):
                continue
            if any((centre[0]-c[0])**2 + (centre[1]-c[1])**2 + (centre[2]-c[2])**2 < SEP2 for c in chosen_centres):
                continue
            chosen.append({"centre": centre, "contrast": float(contrasts[idx]), "band": band_of(float(contrasts[idx]))})
            chosen_centres.append(centre)
    return chosen


def prior_reviewed_edges():
    """All exact edges reviewed in any prior -001..-005 queue, as (left,right,axis)."""
    import glob
    edges = set()
    for qp in glob.glob(str(REPO / "experiments/phase6e/g3-interface-queues-*/MV-G3-*.json")):
        try:
            q = json.loads(Path(qp).read_text())
        except Exception:
            continue
        for question in q.get("questions", []):
            edges.add((tuple(question["pair_left_zyx"]), tuple(question["pair_right_zyx"]), int(question["channel_zyx"])))
    return edges


def _interleave(all_items):
    """Deterministic interleave so axes are not shown in blocks.
    all_items: list of (location_index, axis, edge). Sort by (member position
    in a round-robin of axes, location) to avoid all-Z-then-all-Y-then-all-X."""
    # order key: primary by a rotating axis offset per location, then location
    ordered = sorted(all_items, key=lambda t: ((t[0] + t[1]) % 3, t[0], t[1]))
    return ordered


def build(workspace_root: Path, queue_root: Path, n_crops: int, per_crop: int):
    if workspace_root.exists() or queue_root.exists():
        raise SystemExit("Refusing to overwrite existing -006 artifacts")
    survey = json.loads((REPO / "experiments/phase6e/MV-G3-NEW-REGION-SURVEY-001.json").read_text())
    manifest = json.loads((REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json").read_text())
    src_rec = {r["id"]: r for r in manifest["records"]}
    protocol_005 = json.loads((REPO / "experiments/phase6e/G3_NEW_REGION_SELECTION_PROTOCOL_001.json").read_text())
    used_005 = {s["source_id"] for s in protocol_005["selected"]}

    # Eligible survey candidates NOT used in -005, ranked by spatial gap desc,
    # tie-break source id; take n_crops with distinct y-bands (label-blind).
    pop = {e["candidate_source_id"]: e for e in survey["surveyed_population"]}
    cands = [pop[sid] for sid in survey["eligible_candidates"] if sid not in used_005]
    cands.sort(key=lambda e: (-int(e["min_gap_to_any_existing"]), e["candidate_source_id"]))
    selected = []
    ybands = set()
    for e in cands:
        yb = tuple(e["bounds_xyz"]["y"])
        if yb in ybands:
            continue
        selected.append(e)
        ybands.add(yb)
        if len(selected) >= n_crops:
            break
    if len(selected) < n_crops:
        raise SystemExit(f"Only {len(selected)} fresh spatially-diverse crops available")

    prior_edges = prior_reviewed_edges()
    crop_ids = [f"MV-G3-AXNEU-{chr(ord('A') + i)}" for i in range(len(selected))]

    summary = []
    for crop_id, sel in zip(crop_ids, selected):
        rec = src_rec[sel["candidate_source_id"]]
        arr = np.asarray(np.load(rec["raw_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        locs = select_locations(arr, prior_edges, per_crop)

        crop_dir = workspace_root / crop_id
        crop_dir.mkdir(parents=True)
        log = crop_dir / "workspace.events.jsonl"
        log.write_text("", encoding="utf-8")
        ws_id = f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id}-006"
        workspace = {
            "allowed_decisions": ["BAD_QUESTION", "DIFFERENT_PROCESS", "SAME_PROCESS", "UNCERTAIN"],
            "coordinate_frame": "MV-FRAME-DVID-WASP5-001",
            "created_at": _now(),
            "crop_id": crop_id,
            "event_log": {"append_only": True, "path": str(log.resolve())},
            "id": ws_id,
            "pair_contract": "pair_left_zyx is followed by its positive neighbour along channel_zyx: 0=Z, 1=Y, 2=X.",
            "parent_region_id": sel["candidate_source_id"],
            "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
            "raw": {"path": rec["raw_path"], "sha256": rec["raw_sha256"], "shape_zyx": rec["shape_zyx"], "dtype": rec["dtype"]},
            "provenance": {"sampler": DECLARED_RULE["sampler_id"], "survey": survey["id"]},
            "review_state": "UNREVIEWED",
            "schema_version": 1,
            "source_origin_xyz": [rec["bounds_xyz"][a][0] for a in ("x", "y", "z")],
            "split": "G3_TARGET_TRAIN",
            "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
        }
        (crop_dir / "workspace.json").write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # build all edges, then interleave
        items = []
        for li, loc in enumerate(locs):
            for axis, (left, right) in _edges(loc["centre"]).items():
                items.append((li, axis, (left, right, loc)))
        ordered = _interleave(items)
        questions = []
        for qi, (li, axis, (left, right, loc)) in enumerate(ordered, 1):
            questions.append({
                "id": f"MV-G3-AXNEU-{qi:03d}",
                "kind": "RAW_EM_AXIS_NEUTRAL_EDGE",
                "interface_id": f"MV-G3-AXNEU-LOC{li+1:02d}",  # location groups the triplet for provenance
                "location_index": li + 1,
                "pair_left_zyx": list(left), "pair_right_zyx": list(right), "channel_zyx": axis,
                "axis_name": {0: "Z", 1: "Y", 2: "X"}[axis],
                "location_contrast": loc["contrast"], "contrast_band": loc["band"],
                "raw_gradient_score": loc["contrast"],  # reviewer UI field
                "interface_member": 1, "interface_members": 1,  # each edge is its own independent decision
                "selection": "AXIS_NEUTRAL_PAIRED_LOCATION_V1", "model_navigation": False,
            })
        queue_root.mkdir(parents=True, exist_ok=True)
        queue = {
            "schema_version": 1, "id": f"MV-G3-AXNEU-QUEUE-{crop_id}-006", "created_at": _now(),
            "status": "EXPERT_INTERFACE_REVIEW_REQUIRED", "workspace_id": ws_id, "crop_id": crop_id,
            "raw_sha256": sel["raw_sha256"],
            "selection": {"method": "AXIS_NEUTRAL_PAIRED_LOCATION_V1", "declared_sampling_rule": DECLARED_RULE,
                          "locations": len(locs), "edges_per_location": 3,
                          "band_counts": {b["label"]: sum(1 for L in locs if L["band"] == b["label"]) for b in CONTRAST_BANDS}},
            "questions": questions,
            "scientific_boundary": "Axis-neutral paired-location edges: all three axes are eligible at each center independently of gradient dominance. Each edge is an independent expert decision; stratification targets no class.",
        }
        (queue_root / f"{crop_id}.json").write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
        summary.append({"crop_id": crop_id, "source_id": sel["candidate_source_id"], "workspace_id": ws_id,
                        "locations": len(locs), "edges": len(questions),
                        "band_counts": queue["selection"]["band_counts"]})

    protocol = {"id": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-001", "created_at": _now(),
                "status": "PREDECLARED_FIXED_BEFORE_LABELS", "declared_sampling_rule": DECLARED_RULE,
                "n_crops": n_crops, "locations_per_crop": per_crop, "selected_crops": summary}
    PROTOCOL_OUT.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (queue_root / "batch-manifest.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO / "experiments/phase6e/g3-external-review-packages-006")
    parser.add_argument("--queue-root", type=Path, default=REPO / "experiments/phase6e/g3-interface-queues-006")
    parser.add_argument("--n-crops", type=int, default=3)
    parser.add_argument("--locations-per-crop", type=int, default=LOCATIONS_PER_CROP)
    args = parser.parse_args()
    result = build(args.workspace_root, args.queue_root, args.n_crops, args.locations_per_crop)
    print(json.dumps({"crops": [(s["crop_id"], s["source_id"], s["locations"], s["edges"], s["band_counts"]) for s in result["selected_crops"]]}, indent=2))


if __name__ == "__main__":
    main()
