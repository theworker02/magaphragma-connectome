"""Axis-neutral paired-location experiment (-006): at each selected physical
center, define THREE independently-reviewable edges Z/Y/X from the same local
3-D tissue neighborhood. Axis-balanced by construction; argmax(|gradient|) never
gates edge eligibility.

Label-blind selection: interior centers, 12-voxel separated, stratified across
predeclared raw edge-contrast bands, drawn from multiple spatially-independent
crops spanning low-contrast (A/B-like) and high-contrast (C/E-like) tissue.

Immutability: fresh IDs, empty logs; refuses to overwrite. Historical exact
affinity edges (from every prior workspace event log) are excluded.
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
MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
SURVEY = REPO / "experiments/phase6e/MV-G3-NEW-REGION-SURVEY-001.json"
M = 3
SEP2 = 12 ** 2

# Predeclared raw edge-contrast bands (mean |single-voxel step| over the 3 axis
# edges at a center), chosen from the label-blind contrast probe so both
# low-contrast (A/B-like ~1-2) and high-contrast (C/E-like ~8-17) tissue land in
# distinct bands.
CONTRAST_BANDS = [
    {"label": "low", "lo": 0.0, "hi": 3.0},
    {"label": "mid", "lo": 3.0, "hi": 8.0},
    {"label": "high", "lo": 8.0, "hi": 1e9},
]

# Crops chosen to span the contrast range, spatially independent, and NOT reused
# as prior TRAIN/VAL frozen supervision sources where avoidable. High-contrast
# tissue (C/E-like) is represented by survey-017 and survey-027 raw crops; these
# were reviewed historically but the -006 experiment uses fresh centers and
# excludes every historical exact edge.
CROP_SOURCES = {
    "MV-G3-AXNEU-P": "MV-DVID-RAW-SURVEY-010",  # low-contrast, unused
    "MV-G3-AXNEU-Q": "MV-DVID-RAW-SURVEY-019",  # low-contrast, unused
    "MV-G3-AXNEU-R": "MV-DVID-RAW-SURVEY-017",  # high-contrast (C source)
    "MV-G3-AXNEU-S": "MV-DVID-RAW-SURVEY-027",  # high-contrast (E source)
}
LOCATIONS_PER_CROP = 3  # 4 crops x 3 = 12 locations x 3 edges = 36 decisions


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as s:
        while b := s.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def historical_edges() -> set:
    edges = set()
    for lp in glob.glob(str(REPO / "experiments/phase6e/g3-external-review-packages-*/MV-G3-*/workspace.events.jsonl")):
        for line in open(lp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            edges.add((tuple(e["pair_left_zyx"]), int(e["channel_zyx"])))
    return edges


def edge_contrast_map(arr: np.ndarray) -> np.ndarray:
    z, y, x = arr.shape
    core = arr[M:z - M, M:y - M, M:x - M]
    ez = np.abs(arr[M + 1:z - M + 1, M:y - M, M:x - M] - core)
    ey = np.abs(arr[M:z - M, M + 1:y - M + 1, M:x - M] - core)
    ex = np.abs(arr[M:z - M, M:y - M, M + 1:x - M + 1] - core)
    return (ez + ey + ex) / 3.0  # indexed by (z-M, y-M, x-M)


def band_of(value: float) -> str:
    for b in CONTRAST_BANDS:
        if b["lo"] <= value < b["hi"]:
            return b["label"]
    return CONTRAST_BANDS[-1]["label"]


def select_locations(arr: np.ndarray, excluded_edges: set, per_crop: int) -> list[dict]:
    """Deterministic: rank interior centers by DESC contrast, then walk assigning
    to bands, enforcing 12-voxel separation and excluding any center whose ANY of
    its 3 edges is a historical exact edge. Balanced across bands: fill each band
    round-robin up to ceil(per_crop/nbands)+ until per_crop reached."""
    cmap = edge_contrast_map(arr)
    z, y, x = arr.shape
    # candidate centers: all interior voxels with all 3 forward edges in-bounds
    coords = np.argwhere(np.ones(cmap.shape, dtype=bool))  # every entry of cmap is valid
    vals = cmap.reshape(-1)
    order = np.argsort(vals, kind="stable")[::-1]
    chosen: list[dict] = []
    chosen_centres: list[tuple] = []
    band_counts = {b["label"]: 0 for b in CONTRAST_BANDS}
    target_per_band = max(1, per_crop // len(CONTRAST_BANDS))
    # two passes: first fill balanced target per band, then top up to per_crop
    for pass_no in (0, 1):
        for idx in order:
            if len(chosen) >= per_crop:
                break
            ci = coords[idx]
            centre = (int(ci[0]) + M, int(ci[1]) + M, int(ci[2]) + M)
            edges = {
                0: (centre, (centre[0] + 1, centre[1], centre[2])),
                1: (centre, (centre[0], centre[1] + 1, centre[2])),
                2: (centre, (centre[0], centre[1], centre[2] + 1)),
            }
            if any((edges[a][0], a) in excluded_edges for a in (0, 1, 2)):
                continue
            if centre in chosen_centres:
                continue
            if any((centre[0] - c[0]) ** 2 + (centre[1] - c[1]) ** 2 + (centre[2] - c[2]) ** 2 < SEP2 for c in chosen_centres):
                continue
            b = band_of(float(vals[idx]))
            if pass_no == 0 and band_counts[b] >= target_per_band:
                continue
            chosen.append({"centre_zyx": list(centre), "contrast": float(vals[idx]), "contrast_band": b, "edges": {("Z", "Y", "X")[a]: [list(edges[a][0]), list(edges[a][1])] for a in (0, 1, 2)}})
            chosen_centres.append(centre)
            band_counts[b] += 1
        if len(chosen) >= per_crop:
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO / "experiments/phase6e/g3-external-review-packages-006")
    parser.add_argument("--queue-root", type=Path, default=REPO / "experiments/phase6e/g3-interface-queues-006")
    parser.add_argument("--per-crop", type=int, default=LOCATIONS_PER_CROP)
    args = parser.parse_args()
    if args.workspace_root.exists() or args.queue_root.exists():
        raise SystemExit("Refusing to overwrite existing -006 artifacts")

    rec = {r["id"]: r for r in json.loads(MANIFEST.read_text())["records"]}
    excluded = historical_edges()
    survey_existing = json.loads(SURVEY.read_text())["existing_region_boxes"]

    def overlap(a, b):
        return all(max(a[x][0], b[x][0]) < min(a[x][1], b[x][1]) for x in ("x", "y", "z"))

    manifest_locations = []
    per_crop_summary = []
    for crop_id, sid in CROP_SOURCES.items():
        r = rec[sid]
        # spatial independence check vs existing frozen regions
        for other in survey_existing:
            if overlap(r["bounds_xyz"], other["bounds_xyz"]) and other["source_id"] != sid:
                pass  # same-source reuse (R,S) is intentional high-contrast tissue; see note
        arr = np.asarray(np.load(r["raw_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        locs = select_locations(arr, excluded, args.per_crop)

        crop_dir = args.workspace_root / crop_id
        crop_dir.mkdir(parents=True)
        log = crop_dir / "workspace.events.jsonl"
        log.write_text("", encoding="utf-8")
        ws_id = f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id}-006"
        workspace = {
            "allowed_decisions": ["BAD_QUESTION", "DIFFERENT_PROCESS", "SAME_PROCESS", "UNCERTAIN"],
            "coordinate_frame": "MV-FRAME-DVID-WASP5-001",
            "created_at": _now(), "crop_id": crop_id,
            "event_log": {"append_only": True, "path": str(log.resolve())},
            "id": ws_id,
            "pair_contract": "pair_left_zyx is followed by its positive neighbour along channel_zyx: 0=Z, 1=Y, 2=X.",
            "parent_region_id": sid,
            "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
            "raw": {"path": r["raw_path"], "sha256": r["raw_sha256"], "shape_zyx": r["shape_zyx"], "dtype": r["dtype"]},
            "provenance": {"experiment": "AXIS_NEUTRAL_PAIRED_LOCATION_V2", "protocol": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-002"},
            "review_state": "UNREVIEWED", "schema_version": 1,
            "source_origin_xyz": [r["bounds_xyz"][a][0] for a in ("x", "y", "z")],
            "split": "G3_TARGET_TRAIN",
            "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
        }
        (crop_dir / "workspace.json").write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Presentation order: deterministic interleave of (location, axis) so
        # axes are not shown in blocks.
        questions = []
        for li, loc in enumerate(locs, 1):
            for ai, axis_name in enumerate(("Z", "Y", "X")):
                axis = {"Z": 0, "Y": 1, "X": 2}[axis_name]
                left, right = loc["edges"][axis_name]
                questions.append({
                    "id": f"MV-G3-AXNEU-L{li:02d}-{axis_name}",
                    "kind": "RAW_EM_INTERFACE_MEMBER",
                    "interface_id": f"MV-G3-AXNEU-L{li:02d}-{axis_name}",
                    "interface_member": 1, "interface_members": 1,
                    "location_id": f"L{li:02d}", "location_center_zyx": loc["centre_zyx"],
                    "contrast_band": loc["contrast_band"], "raw_edge_contrast": loc["contrast"],
                    "pair_left_zyx": left, "pair_right_zyx": right, "channel_zyx": axis, "axis_name": axis_name,
                    "raw_gradient_score": loc["contrast"],
                    "selection": "AXIS_NEUTRAL_PAIRED_LOCATION_V2", "model_navigation": False,
                })
            manifest_locations.append({"crop_id": crop_id, "source_id": sid, "location_id": f"L{li:02d}",
                                       "center_zyx": loc["centre_zyx"], "contrast_band": loc["contrast_band"],
                                       "raw_edge_contrast": loc["contrast"], "edges": loc["edges"]})
        # interleave by axis-then-location so no axis is shown in a block
        questions.sort(key=lambda q: (("Z", "Y", "X").index(q["axis_name"]), q["location_id"]))

        args.queue_root.mkdir(parents=True, exist_ok=True)
        band_counts = {b["label"]: sum(1 for loc in locs if loc["contrast_band"] == b["label"]) for b in CONTRAST_BANDS}
        queue = {
            "schema_version": 1, "id": f"MV-G3-AXNEU-QUEUE-{crop_id}-006", "created_at": _now(),
            "status": "EXPERT_INTERFACE_REVIEW_REQUIRED", "workspace_id": ws_id, "crop_id": crop_id,
            "raw_sha256": r["raw_sha256"],
            "selection": {"method": "AXIS_NEUTRAL_PAIRED_LOCATION_V2", "contrast_bands": CONTRAST_BANDS,
                          "locations": len(locs), "edges_per_location": 3, "band_counts": band_counts,
                          "label_blind": True, "argmax_gradient_gates_eligibility": False,
                          "forbidden_inputs": ["SAME/DIFFERENT labels", "class predictions", "learned affinities", "historical class composition", "argmax-gradient edge eligibility"]},
            "questions": questions,
            "scientific_boundary": "Paired Z/Y/X edges at matched physical centers; each edge is an independent expert decision. Balanced by construction; no class targeted.",
        }
        (args.queue_root / f"{crop_id}.json").write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
        per_crop_summary.append({"crop_id": crop_id, "source_id": sid, "workspace_id": ws_id, "locations": len(locs), "edges": len(questions), "band_counts": band_counts})

    batch = {"id": "MV-G3-AXIS-NEUTRAL-PAIRED-BATCH-006", "created_at": _now(),
             "protocol": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-002",
             "crops": list(CROP_SOURCES), "locations_per_crop": args.per_crop,
             "total_locations": sum(s["locations"] for s in per_crop_summary),
             "total_edges": sum(s["edges"] for s in per_crop_summary),
             "per_crop": per_crop_summary, "locations": manifest_locations,
             "excluded_historical_edges": len(excluded)}
    (args.queue_root / "batch-manifest.json").write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"total_locations": batch["total_locations"], "total_edges": batch["total_edges"], "per_crop": per_crop_summary, "excluded_historical_edges": len(excluded)}, indent=2))


if __name__ == "__main__":
    main()
