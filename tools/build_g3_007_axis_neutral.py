"""Protocol-003 + -007: axis-neutral paired-location experiment (clean version).

Design (predeclared, fixed before labels):
  * 12 fresh eligible crops in CANONICAL source-id order (label-blind ordering).
  * Fixed 4-stratum rotation by position: [VERY_LOW, LOW, MID, HIGH] repeating,
    so crop i (0-based) is assigned band ROTATION[i % 4]. Crop order and band
    assignment are fixed independently of any contrast statistic or label.
  * 1 physical location per crop, chosen from interior, 12-voxel spatially
    separated candidate centers whose local raw edge contrast falls in the
    crop's assigned band, selected by a STABLE HASH of immutable identifiers
    (source_id + voxel coordinate) -- NOT by "highest contrast in band".
  * 3 independently reviewable edges per location: Z (z<->z+1), Y (y<->y+1),
    X (x<->x+1). Axis-balanced by construction; argmax(|gradient|) never gates
    edge eligibility.
  * = 12 locations = 36 edges.

Label-blind: contrast used ONLY to place a location in its assigned band; no
SAME/DIFFERENT label, prediction, or learned affinity is consulted. Excludes
every exact affinity edge already present in ANY historical queue/event log
(including the 9 partially-reviewed -006 AXNEU-A events -- coordinates only,
decisions never inspected).
"""
from __future__ import annotations

import argparse
import glob
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
N_CROPS = 12
LOCATIONS_PER_CROP = 1
ROTATION = ["VERY_LOW", "LOW", "MID", "HIGH"]
# Raw edge-contrast bands in mean |single-voxel intensity step| units.
BANDS = {"VERY_LOW": (0.0, 3.0), "LOW": (3.0, 8.0), "MID": (8.0, 20.0), "HIGH": (20.0, 1e9)}

DECLARED_RULE = {
    "sampler_id": "AXIS_NEUTRAL_PAIRED_LOCATION_FIXED_ROTATION_V2",
    "declared_before_labels": True,
    "crop_ordering": "canonical ascending source_id over eligible survey crops; label-blind and contrast-blind",
    "stratum_rotation": ROTATION,
    "stratum_assignment": "crop at 0-based position i -> ROTATION[i % 4] (fixed; independent of that crop's contrast)",
    "contrast_bands_mean_abs_step": BANDS,
    "location_contrast_metric": "mean |c->c+1| intensity step over the location's three axis edges (raw EM only)",
    "within_band_location_pick": "among interior, 12-voxel-separated centers whose contrast is in the assigned band, choose the minimal stable BLAKE2b hash of f'{source_id}:{z},{y},{x}' -- reproducible, not a biological optimum",
    "edges": {"Z": "(z,y,x)<->(z+1,y,x)", "Y": "(z,y,x)<->(z,y+1,x)", "X": "(z,y,x)<->(z,y,x+1)"},
    "edge_eligibility": "all three axis edges eligible independently at each location; argmax(|gradient|) never gates eligibility",
    "forbidden_inputs": ["SAME/DIFFERENT labels", "class predictions", "learned affinities", "historical class composition", "argmax-gradient edge eligibility", "contrast-sorted crop assignment"],
    "excludes": "every exact affinity edge in any historical queue or event log (coordinates only; -006 AXNEU-A decisions NOT inspected)",
}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _stable_hash(source_id: str, z: int, y: int, x: int) -> str:
    return hashlib.blake2b(f"{source_id}:{z},{y},{x}".encode(), digest_size=16).hexdigest()


def historical_excluded_edges() -> set:
    """Every exact (left,right,axis) edge already asked or reviewed anywhere.
    Reads queue questions and event-log pair coordinates ONLY (no decisions)."""
    edges = set()
    for qp in glob.glob(str(REPO / "experiments/phase6e/g3-interface-queues-*/*.json")):
        try:
            data = json.loads(Path(qp).read_text())
        except Exception:
            continue
        for q in data.get("questions", []):
            edges.add((tuple(q["pair_left_zyx"]), tuple(q["pair_right_zyx"]), int(q["channel_zyx"])))
    for lp in glob.glob(str(REPO / "experiments/phase6e/g3-external-review-packages-*/MV-G3-*/workspace.events.jsonl")):
        for line in open(lp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            # coordinates + channel only; decision intentionally ignored
            edges.add((tuple(e["pair_left_zyx"]), tuple(e["pair_right_zyx"]), int(e["channel_zyx"])))
    return edges


def _edges_for_center(c):
    z, y, x = c
    return {
        "Z": ((z, y, x), (z + 1, y, x), 0),
        "Y": ((z, y, x), (z, y + 1, x), 1),
        "X": ((z, y, x), (z, y, x + 1), 2),
    }


def local_edge_contrast(arr, c):
    z, y, x = c
    ez = abs(float(arr[z + 1, y, x]) - float(arr[z, y, x]))
    ey = abs(float(arr[z, y + 1, x]) - float(arr[z, y, x]))
    ex = abs(float(arr[z, y, x + 1]) - float(arr[z, y, x]))
    return (ez + ey + ex) / 3.0


def separated_interior_centers(arr):
    """Deterministic set of interior, 12-voxel separated candidate centers.
    Walk voxels in raster order (label-blind, contrast-blind ordering) and
    greedily keep separated ones. Returns list of (z,y,x)."""
    z, y, x = arr.shape
    kept = []
    sep = 12
    grid = {}
    for zz in range(MARGIN, z - MARGIN):
        for yy in range(MARGIN, y - MARGIN):
            for xx in range(MARGIN, x - MARGIN):
                cell = (zz // sep, yy // sep, xx // sep)
                conflict = False
                for dz in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            for (az, ay, ax) in grid.get((cell[0]+dz, cell[1]+dy, cell[2]+dx), ()):
                                if (az-zz)**2 + (ay-yy)**2 + (ax-xx)**2 < SEP2:
                                    conflict = True
                                    break
                            if conflict: break
                        if conflict: break
                    if conflict: break
                if not conflict:
                    grid.setdefault(cell, []).append((zz, yy, xx))
                    kept.append((zz, yy, xx))
    return kept


def build(protocol_out: Path, ws_root: Path, q_root: Path):
    survey = json.loads(SURVEY.read_text())
    manifest = json.loads(MANIFEST.read_text())
    rec = {r["id"]: r for r in manifest["records"]}
    eligible = sorted(survey["eligible_candidates"])  # canonical source-id order
    if len(eligible) < N_CROPS:
        raise SystemExit(f"Need {N_CROPS} eligible crops, have {len(eligible)}")
    chosen_sources = eligible[:N_CROPS]
    excluded_edges = historical_excluded_edges()

    selected = []
    for i, sid in enumerate(chosen_sources):
        band = ROTATION[i % 4]
        lo, hi = BANDS[band]
        arr = np.asarray(np.load(rec[sid]["raw_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        centers = separated_interior_centers(arr)
        # candidates in the assigned band whose 3 edges are all fresh + in-bounds
        cands = []
        z, y, x = arr.shape
        for c in centers:
            if not (c[0] + 1 < z and c[1] + 1 < y and c[2] + 1 < x):
                continue
            contrast = local_edge_contrast(arr, c)
            if not (lo <= contrast < hi):
                continue
            edges = _edges_for_center(c)
            if any((e[0], e[1], e[2]) in excluded_edges for e in edges.values()):
                continue
            cands.append((c, contrast))
        if not cands:
            raise SystemExit(f"{sid}: no fresh interior candidate in band {band}")
        # deterministic pick: minimal stable hash
        pick = min(cands, key=lambda cc: _stable_hash(sid, cc[0][0], cc[0][1], cc[0][2]))
        c, contrast = pick
        selected.append({
            "crop_index": i, "assigned_band": band, "source_id": sid,
            "crop_id": f"MV-G3-AXNEU2-{i+1:02d}",
            "center_zyx": list(c), "local_edge_contrast": round(contrast, 4),
            "selection_hash": _stable_hash(sid, c[0], c[1], c[2]),
            "raw_sha256": rec[sid]["raw_sha256"], "raw_path": rec[sid]["raw_path"],
            "shape_zyx": rec[sid]["shape_zyx"], "bounds_xyz": rec[sid]["bounds_xyz"],
            "edges": {k: {"left": list(v[0]), "right": list(v[1]), "axis": v[2]} for k, v in _edges_for_center(c).items()},
        })

    protocol = {
        "id": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-003",
        "schema_version": 1, "created_at": _now(),
        "status": "PREDECLARED_FIXED_BEFORE_LABELS",
        "declared_sampling_rule": DECLARED_RULE,
        "n_crops": N_CROPS, "locations_per_crop": LOCATIONS_PER_CROP,
        "total_locations": len(selected), "total_edges": 3 * len(selected),
        "survey": {"path": str(SURVEY.resolve())},
        "excluded_historical_edge_count": len(excluded_edges),
        "selected_locations": selected,
    }
    if protocol_out:
        protocol_out.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Build -007 workspaces + queues (only when writing to the real roots)
    if ws_root and q_root:
        if ws_root.exists() or q_root.exists():
            raise SystemExit("Refusing to overwrite existing -007 artifacts")
        q_root.mkdir(parents=True)
        for loc in selected:
            crop_id = loc["crop_id"]
            crop_dir = ws_root / crop_id
            crop_dir.mkdir(parents=True)
            log = crop_dir / "workspace.events.jsonl"
            log.write_text("", encoding="utf-8")
            ws_id = f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id}-007"
            workspace = {
                "allowed_decisions": ["BAD_QUESTION", "DIFFERENT_PROCESS", "SAME_PROCESS", "UNCERTAIN"],
                "coordinate_frame": "MV-FRAME-DVID-WASP5-001", "created_at": _now(),
                "crop_id": crop_id, "event_log": {"append_only": True, "path": str(log.resolve())},
                "id": ws_id, "pair_contract": "pair_left_zyx is followed by its positive neighbour along channel_zyx: 0=Z, 1=Y, 2=X.",
                "parent_region_id": loc["source_id"],
                "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
                "raw": {"path": loc["raw_path"], "sha256": loc["raw_sha256"], "shape_zyx": loc["shape_zyx"], "dtype": "uint8"},
                "provenance": {"protocol": {"path": str(protocol_out.resolve()) if protocol_out else None, "id": protocol["id"]},
                               "assigned_band": loc["assigned_band"], "center_zyx": loc["center_zyx"]},
                "review_state": "UNREVIEWED", "schema_version": 1,
                "source_origin_xyz": [loc["bounds_xyz"][a][0] for a in ("x", "y", "z")],
                "split": "G3_TARGET_TRAIN", "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
            }
            (crop_dir / "workspace.json").write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            # 3 edges -> single-member interfaces each (independent decisions)
            questions = []
            for idx, axis_name in enumerate(("Z", "Y", "X"), 1):
                e = loc["edges"][axis_name]
                questions.append({
                    "id": f"MV-G3-IF-{idx:03d}-01", "kind": "RAW_EM_INTERFACE_MEMBER",
                    "interface_id": f"MV-G3-IF-{idx:03d}", "interface_member": 1, "interface_members": 1,
                    "pair_left_zyx": e["left"], "pair_right_zyx": e["right"], "channel_zyx": e["axis"],
                    "axis_name": axis_name, "assigned_band": loc["assigned_band"],
                    "raw_gradient_score": 0.0, "selection": DECLARED_RULE["sampler_id"], "model_navigation": False,
                })
            queue = {
                "schema_version": 1, "id": f"MV-G3-AXNEU2-QUEUE-{crop_id}-007", "created_at": _now(),
                "status": "EXPERT_INTERFACE_REVIEW_REQUIRED", "workspace_id": ws_id, "crop_id": crop_id,
                "raw_sha256": loc["raw_sha256"],
                "selection": {"method": DECLARED_RULE["sampler_id"], "declared_sampling_rule": DECLARED_RULE,
                              "assigned_band": loc["assigned_band"], "center_zyx": loc["center_zyx"]},
                "questions": questions,
                "scientific_boundary": "Axis-neutral paired-location edges at one physical center; each edge is an independent expert decision. Balanced Z/Y/X by construction; no class targeted.",
            }
            (q_root / f"{crop_id}.json").write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")

    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol-out", type=Path, default=REPO / "experiments/phase6e/G3_AXIS_NEUTRAL_SAMPLING_PROTOCOL_003.json")
    p.add_argument("--workspace-root", type=Path, default=REPO / "experiments/phase6e/g3-external-review-packages-007")
    p.add_argument("--queue-root", type=Path, default=REPO / "experiments/phase6e/g3-interface-queues-007")
    args = p.parse_args()
    proto = build(args.protocol_out, args.workspace_root, args.queue_root)
    print(json.dumps({"locations": proto["total_locations"], "edges": proto["total_edges"],
                      "crops": [(s["crop_id"], s["source_id"], s["assigned_band"], s["center_zyx"], s["local_edge_contrast"]) for s in proto["selected_locations"]]}, indent=2))


if __name__ == "__main__":
    main()
