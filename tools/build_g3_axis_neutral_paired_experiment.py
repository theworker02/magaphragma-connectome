"""Axis-neutral paired-location experiment (V2): at each selected physical
center, present all THREE axis edges (Z,Y,X) as separately reviewable
interfaces from the SAME local tissue neighborhood. This controls crop and
local morphology so we can test whether the historical Z=DIFFERENT / Y,X=SAME
pattern survives when location is held fixed.

Design (predeclared, label-blind, fixed BEFORE any label):
  * Location center c=(z,y,x), interior (all three c+1 neighbors in-bounds).
  * Three edges per location: Z (c<->c+1z), Y (c<->c+1y), X (c<->c+1x).
    Edge eligibility is INDEPENDENT of argmax(|gradient|); all three are always
    generated. This is the core H3-vs-H4 discriminator.
  * Location stratifier (raw EM only): local edge contrast =
    mean(|c->c+1z|, |c->c+1y|, |c->c+1x|).
  * Contrast bands (declared from a read-only probe of the raw pool, NOT from
    labels): LOW [0,4), MID [4,10), HIGH [10, inf).
  * Crops: multiple spatially-independent crops spanning BOTH low-contrast
    (A/010/014/019-like) and high-contrast (C=017, E=027-like) tissue, so
    contrast strata are populated by construction.
  * Locations chosen deterministically: within each (crop, target-band), walk
    interior centers in a fixed raster order, keep those whose contrast falls in
    the band, enforce 12-voxel separation, take the required count.
  * Exclude any location whose ANY axis edge exactly matches a historical
    reviewed edge (-001..-005).
  * Target ~10 locations => 30 edges, balanced Z/Y/X by construction (10 each).

Selection uses only raw EM intensity/gradient-for-contrast, crop identity, and
geometry. No SAME/DIFFERENT label, prediction, or learned affinity is consulted.
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
SURVEY_MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
# v2 experiment written to distinct roots so the pre-existing PROTOCOL-001
# -006 batch (crops AXNEU-A/B/C, partially reviewed) is preserved untouched.
WS_ROOT = REPO / "experiments/phase6e/g3-external-review-packages-006b"
Q_ROOT = REPO / "experiments/phase6e/g3-interface-queues-006b"
PROTOCOL_OUT = REPO / "experiments/phase6e/G3_AXIS_NEUTRAL_SAMPLING_PROTOCOL_002.json"
MARGIN = 3
SEP2 = 12 ** 2

# Crops chosen for contrast diversity (label-blind: contrast is raw EM only).
# Low-contrast tissue + high-contrast tissue (017=C-source, 027=E-source).
CROPS = [
    {"crop_id": "MV-G3-AXNEU2-LO1", "source_id": "MV-DVID-RAW-SURVEY-010", "regime": "low"},
    {"crop_id": "MV-G3-AXNEU2-LO2", "source_id": "MV-DVID-RAW-SURVEY-019", "regime": "low"},
    {"crop_id": "MV-G3-AXNEU2-HI1", "source_id": "MV-DVID-RAW-SURVEY-017", "regime": "high"},
    {"crop_id": "MV-G3-AXNEU2-HI2", "source_id": "MV-DVID-RAW-SURVEY-027", "regime": "high"},
]
CONTRAST_BANDS = [("LOW", 0.0, 4.0), ("MID", 4.0, 10.0), ("HIGH", 10.0, float("inf"))]
# Deterministic target: for each crop pick locations to reach ~10-12 total.
# low-regime crops contribute LOW/MID; high-regime crops contribute MID/HIGH.
LOCATIONS_PER_CROP = 3  # 4 crops x 3 = 12 locations => 36 edges


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
            edges.add((tuple(e["pair_left_zyx"]), tuple(e["pair_right_zyx"]), int(e["channel_zyx"])))
    return edges


def local_contrast(arr, c):
    z, y, x = c
    base = float(arr[z, y, x])
    return (abs(float(arr[z + 1, y, x]) - base) + abs(float(arr[z, y + 1, x]) - base) + abs(float(arr[z, y, x + 1]) - base)) / 3.0


def three_edges(c):
    z, y, x = c
    return [
        ((z, y, x), (z + 1, y, x), 0),
        ((z, y, x), (z, y + 1, x), 1),
        ((z, y, x), (z, y, x + 1), 2),
    ]


def select_locations(arr, excluded_edges, want):
    """Deterministic raster walk; keep interior 12-sep centers, stratify by band."""
    z, y, x = arr.shape
    chosen = []
    chosen_centres = []
    band_counts = {b[0]: 0 for b in CONTRAST_BANDS}
    # deterministic raster order
    for zz in range(MARGIN, z - MARGIN - 1):
        for yy in range(MARGIN, y - MARGIN - 1):
            for xx in range(MARGIN, x - MARGIN - 1):
                if len(chosen) >= want:
                    return chosen, band_counts
                c = (zz, yy, xx)
                edges = three_edges(c)
                if any((e[0], e[1], e[2]) in excluded_edges for e in edges):
                    continue
                if any((c[0] - o[0]) ** 2 + (c[1] - o[1]) ** 2 + (c[2] - o[2]) ** 2 < SEP2 for o in chosen_centres):
                    continue
                contrast = local_contrast(arr, c)
                band = next(b[0] for b in CONTRAST_BANDS if b[1] <= contrast < b[2])
                chosen.append({"center_zyx": list(c), "contrast": round(contrast, 3), "contrast_band": band})
                chosen_centres.append(c)
                band_counts[band] += 1
    return chosen, band_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations-per-crop", type=int, default=LOCATIONS_PER_CROP)
    args = parser.parse_args()
    if WS_ROOT.exists() or Q_ROOT.exists() or PROTOCOL_OUT.exists():
        raise SystemExit("Refusing to overwrite existing -006 / protocol-002 artifacts")
    manifest = json.loads(SURVEY_MANIFEST.read_text())
    rec = {r["id"]: r for r in manifest["records"]}
    excluded = historical_edges()

    crop_summaries = []
    for crop in CROPS:
        r = rec[crop["source_id"]]
        arr = np.asarray(np.load(r["raw_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        locs, band_counts = select_locations(arr, excluded, args.locations_per_crop)
        if len(locs) < args.locations_per_crop:
            raise SystemExit(f"{crop['crop_id']}: only {len(locs)} eligible locations")

        crop_dir = WS_ROOT / crop["crop_id"]
        crop_dir.mkdir(parents=True)
        log = crop_dir / "workspace.events.jsonl"
        log.write_text("", encoding="utf-8")
        ws_id = f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop['crop_id']}-006"
        workspace = {
            "allowed_decisions": ["BAD_QUESTION", "DIFFERENT_PROCESS", "SAME_PROCESS", "UNCERTAIN"],
            "coordinate_frame": "MV-FRAME-DVID-WASP5-001",
            "created_at": _now(), "crop_id": crop["crop_id"],
            "event_log": {"append_only": True, "path": str(log.resolve())},
            "id": ws_id,
            "pair_contract": "pair_left_zyx is followed by its positive neighbour along channel_zyx: 0=Z, 1=Y, 2=X.",
            "parent_region_id": crop["source_id"],
            "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
            "raw": {"path": r["raw_path"], "sha256": r["raw_sha256"], "shape_zyx": r["shape_zyx"], "dtype": r["dtype"]},
            "provenance": {"experiment": "AXIS_NEUTRAL_PAIRED_LOCATION_V2", "contrast_regime": crop["regime"]},
            "review_state": "UNREVIEWED", "schema_version": 1,
            "source_origin_xyz": [r["bounds_xyz"][a][0] for a in ("x", "y", "z")],
            "split": "G3_DIAGNOSTIC_AXIS_NEUTRAL", "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
        }
        (crop_dir / "workspace.json").write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Deterministic interleave: present (location, axis) so axes not blocked.
        questions = []
        AX = {0: "Z", 1: "Y", 2: "X"}
        for li, loc in enumerate(locs, 1):
            for (left, right, axis) in three_edges(tuple(loc["center_zyx"])):
                questions.append({
                    "id": f"MV-G3-AXNEU-L{li:02d}-{AX[axis]}",
                    "kind": "RAW_EM_AXIS_NEUTRAL_PAIRED_EDGE",
                    "interface_id": f"MV-G3-AXNEU-L{li:02d}-{AX[axis]}",
                    "interface_member": 1, "interface_members": 1,
                    "location_id": f"L{li:02d}", "location_center_zyx": loc["center_zyx"],
                    "pair_left_zyx": list(left), "pair_right_zyx": list(right), "channel_zyx": axis, "axis_name": AX[axis],
                    "contrast_band": loc["contrast_band"], "local_contrast": loc["contrast"],
                    "raw_gradient_score": loc["contrast"],  # reviewer displays this field
                    "selection": "AXIS_NEUTRAL_PAIRED_LOCATION_V2", "model_navigation": False,
                })
        # interleave by axis position within each location is already (L,Z),(L,Y),(L,X)
        Q_ROOT.mkdir(parents=True, exist_ok=True)
        queue = {
            "schema_version": 1, "id": f"MV-G3-AXNEU-QUEUE-{crop['crop_id']}-006", "created_at": _now(),
            "status": "EXPERT_INTERFACE_REVIEW_REQUIRED", "workspace_id": ws_id, "crop_id": crop["crop_id"],
            "raw_sha256": r["raw_sha256"],
            "selection": {"method": "AXIS_NEUTRAL_PAIRED_LOCATION_V2", "contrast_regime": crop["regime"],
                          "band_counts": band_counts, "locations": len(locs),
                          "edge_eligibility": "all three axis edges eligible independently; NOT argmax-gated",
                          "label_blind": True,
                          "forbidden_inputs": ["SAME/DIFFERENT labels", "class predictions", "learned affinities", "historical class composition", "argmax-gradient edge eligibility"],
                          "excluded_historical_edges": len(excluded)},
            "locations": locs, "questions": questions,
            "scientific_boundary": "Axis-neutral paired-location diagnostic: at each fixed physical center all three axis edges are reviewed independently. Purpose is to test axis/class confounding; not training supervision.",
        }
        (Q_ROOT / f"{crop['crop_id']}.json").write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
        crop_summaries.append({"crop_id": crop["crop_id"], "source_id": crop["source_id"], "regime": crop["regime"],
                               "workspace_id": ws_id, "locations": len(locs), "edges": 3 * len(locs), "band_counts": band_counts})

    protocol = {
        "id": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-002", "schema_version": 1,
        "status": "PREDECLARED_FIXED_BEFORE_LABELS", "created_at": _now(),
        "supersedes_note": "New version; does not modify PROTOCOL-001 (preserved).",
        "experiment": "AXIS_NEUTRAL_PAIRED_LOCATION_V2",
        "design": {
            "edges_per_location": {"Z": "(z,y,x)<->(z+1,y,x)", "Y": "(z,y,x)<->(z,y+1,x)", "X": "(z,y,x)<->(z,y,x+1)"},
            "edge_eligibility": "all three eligible independently; NOT argmax(|gradient|) gated",
            "location_stratifier": "mean |c->c+1| over the three axis edges (raw EM only)",
            "contrast_bands": [{"label": b[0], "lo": b[1], "hi": (None if b[2] == float('inf') else b[2])} for b in CONTRAST_BANDS],
            "crops": CROPS, "locations_per_crop": args.locations_per_crop,
            "total_locations": args.locations_per_crop * len(CROPS), "total_edges": 3 * args.locations_per_crop * len(CROPS),
            "axis_balance_by_construction": "exactly one Z, one Y, one X edge per location",
        },
        "survey_manifest": {"path": str(SURVEY_MANIFEST.resolve()), "sha256": _sha(SURVEY_MANIFEST)},
        "per_crop": crop_summaries,
    }
    PROTOCOL_OUT.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(PROTOCOL_OUT.resolve()), "per_crop": crop_summaries,
                      "total_locations": protocol["design"]["total_locations"], "total_edges": protocol["design"]["total_edges"]}, indent=2))


if __name__ == "__main__":
    main()
