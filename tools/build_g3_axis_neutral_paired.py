"""Axis-neutral PAIRED-LOCATION experiment (protocol/version -002, batch -006).

Goal: separate H3 (dominant-gradient selection) from H4 (intrinsic Z-edge
biology) by reviewing ALL THREE axis edges (Z,Y,X) at the SAME physical center,
across multiple spatially-independent crops spanning predeclared raw-contrast
strata. Holding crop + local tissue neighborhood constant per location removes
the crop/orientation confound that the historical geometry could not.

Design (predeclared, label-blind, fixed before any -006 label):
  * crops: 2 high-contrast (C/E-like) + 2 low-contrast, chosen by a fixed list
    below (raw-contrast is a raw-EM statistic, not a label);
  * per crop: 3 interior centers, 12-voxel separated, deterministically drawn
    to spread across the location's raw edge-contrast stratum;
  * per center: exactly 3 edges Z:(z,y,x)-(z+1,y,x), Y:-(z,y+1,x), X:-(z,y,x+1);
  * edge eligibility is INDEPENDENT per axis and does NOT use argmax(|grad|);
  * exclude any exact affinity edge already present in ANY historical queue;
  * axis-balanced by construction (equal Z/Y/X counts);
  * presentation interleaves (location,axis) so axes are not shown in blocks.

No SAME/DIFFERENT label, prediction, or learned affinity is consulted.
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
PROTOCOL_OUT = REPO / "experiments/phase6e/G3_AXIS_NEUTRAL_SAMPLING_PROTOCOL_002.json"
MARGIN = 3
SEP2 = 12 ** 2
LOCATIONS_PER_CROP = 3

# Predeclared crop list: raw-contrast strata by raw-EM statistics only.
# high-contrast tissue (C/E-like) + low-contrast (A-like). These are DIAGNOSTIC
# locations; C/E remain training-rejected per the admission decision.
CROPS = [
    {"crop_id": "MV-G3-AXNEU2-HC1", "source_id": "MV-DVID-RAW-SURVEY-017", "contrast_class": "high"},
    {"crop_id": "MV-G3-AXNEU2-HC2", "source_id": "MV-DVID-RAW-SURVEY-027", "contrast_class": "high"},
    {"crop_id": "MV-G3-AXNEU2-LC1", "source_id": "MV-DVID-RAW-SURVEY-010", "contrast_class": "low"},
    {"crop_id": "MV-G3-AXNEU2-LC2", "source_id": "MV-DVID-RAW-SURVEY-019", "contrast_class": "low"},
]

DECLARED_RULE = {
    "sampler_id": "AXIS_NEUTRAL_PAIRED_LOCATION_V2",
    "declared_before_labels": True,
    "edges": {"Z": "(z,y,x)<->(z+1,y,x)", "Y": "(z,y,x)<->(z,y+1,x)", "X": "(z,y,x)<->(z,y,x+1)"},
    "edge_eligibility": "all three edges eligible INDEPENDENTLY at each center; NOT gated by argmax(|gradient|)",
    "location_selection": "interior centers (>=3 from every face so all 3 forward edges in-bounds), 12-voxel separated, ranked by a fixed contrast key and drawn to spread across the crop's local edge-contrast quantiles",
    "location_contrast_key": "mean of |c->c+1| raw intensity step across the three axis edges at c (raw EM only)",
    "locations_per_crop": LOCATIONS_PER_CROP,
    "crops": CROPS,
    "axis_balanced_by_construction": True,
    "presentation_order": "deterministic interleave of (location, axis)",
    "exclusions": "exact affinity edges present in ANY historical queue (-001..-005)",
    "forbidden_inputs": ["argmax-gradient edge eligibility", "SAME/DIFFERENT labels", "class predictions", "learned affinities", "historical class composition"],
    "purpose": "H3 vs H4 discrimination with crop + local tissue held constant per location",
}


def _now():
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as s:
        while b := s.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def historical_edges() -> set:
    edges = set()
    for qp in glob.glob(str(REPO / "experiments/phase6e/g3-interface-queues-*/*.json")):
        if Path(qp).name == "batch-manifest.json":
            continue
        try:
            q = json.loads(Path(qp).read_text())
        except Exception:
            continue
        for qq in q.get("questions", []):
            edges.add((tuple(qq["pair_left_zyx"]), tuple(qq["pair_right_zyx"]), int(qq["channel_zyx"])))
    return edges


def location_contrast(arr, c):
    z, y, x = c
    return float((abs(arr[z+1, y, x]-arr[z, y, x]) + abs(arr[z, y+1, x]-arr[z, y, x]) + abs(arr[z, y, x+1]-arr[z, y, x])) / 3.0)


def _edges_for(cc):
    return [
        (cc, (cc[0] + 1, cc[1], cc[2]), 0),
        (cc, (cc[0], cc[1] + 1, cc[2]), 1),
        (cc, (cc[0], cc[1], cc[2] + 1), 2),
    ]


def select_locations(arr, n, excluded):
    """Deterministic, argmax-free selection of n interior centers.

    All interior centers where the 3 forward edges are in-bounds are ranked by
    ascending contrast key. We then pick n anchors evenly across the sorted
    (quantile) positions and, from each anchor, walk outward to the nearest
    center that is 12-voxel separated from already-chosen centers and whose
    edges are not historical exact edges. This spreads locations across the
    crop's raw edge-contrast distribution without any label input.
    """
    z, y, x = arr.shape
    zs, ys, xs = np.meshgrid(
        np.arange(MARGIN, z - MARGIN - 1), np.arange(MARGIN, y - MARGIN - 1), np.arange(MARGIN, x - MARGIN - 1), indexing="ij"
    )
    centers = np.stack([zs.ravel(), ys.ravel(), xs.ravel()], axis=1)
    core = arr[MARGIN:z - MARGIN - 1, MARGIN:y - MARGIN - 1, MARGIN:x - MARGIN - 1]
    ez = np.abs(arr[MARGIN + 1:z - MARGIN, MARGIN:y - MARGIN - 1, MARGIN:x - MARGIN - 1] - core)
    ey = np.abs(arr[MARGIN:z - MARGIN - 1, MARGIN + 1:y - MARGIN, MARGIN:x - MARGIN - 1] - core)
    ex = np.abs(arr[MARGIN:z - MARGIN - 1, MARGIN:y - MARGIN - 1, MARGIN + 1:x - MARGIN] - core)
    key = ((ez + ey + ex) / 3.0).ravel()
    order = np.argsort(key, kind="stable")  # ascending contrast
    ncand = len(order)
    anchors = [int((i + 0.5) / n * ncand) for i in range(n)]

    chosen: list[dict] = []
    chosen_centers: list[tuple] = []

    def try_pick(rank_pos: int) -> bool:
        cc = tuple(int(v) for v in centers[order[rank_pos]])
        edges = _edges_for(cc)
        if any((e[0], e[1], e[2]) in excluded for e in edges):
            return False
        if any((cc[0] - o[0]) ** 2 + (cc[1] - o[1]) ** 2 + (cc[2] - o[2]) ** 2 < SEP2 for o in chosen_centers):
            return False
        chosen.append({"center_zyx": list(cc), "contrast_key": location_contrast(arr, cc), "edges": edges})
        chosen_centers.append(cc)
        return True

    for a in anchors:
        placed = False
        for delta in range(ncand):
            for pos in (a + delta, a - delta):
                if 0 <= pos < ncand and try_pick(pos):
                    placed = True
                    break
            if placed:
                break
    return chosen[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO / "experiments/phase6e/g3-external-review-packages-006")
    parser.add_argument("--queue-root", type=Path, default=REPO / "experiments/phase6e/g3-interface-queues-006")
    args = parser.parse_args()
    if args.workspace_root.exists() or args.queue_root.exists() or PROTOCOL_OUT.exists():
        raise SystemExit("Refusing to overwrite existing -006 / protocol-002 artifacts")
    manifest = json.loads(MANIFEST.read_text())
    rec = {r["id"]: r for r in manifest["records"]}
    excluded = historical_edges()

    selected = []
    for crop in CROPS:
        r = rec[crop["source_id"]]
        arr = np.asarray(np.load(r["raw_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        locs = select_locations(arr, LOCATIONS_PER_CROP, excluded)
        crop_dir = args.workspace_root / crop["crop_id"]
        crop_dir.mkdir(parents=True)
        log = crop_dir / "workspace.events.jsonl"
        log.write_text("", encoding="utf-8")
        ws_id = f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop['crop_id']}-006"
        workspace = {
            "allowed_decisions": ["BAD_QUESTION", "DIFFERENT_PROCESS", "SAME_PROCESS", "UNCERTAIN"],
            "coordinate_frame": "MV-FRAME-DVID-WASP5-001", "created_at": _now(),
            "crop_id": crop["crop_id"], "event_log": {"append_only": True, "path": str(log.resolve())},
            "id": ws_id, "pair_contract": "pair_left_zyx followed by positive neighbour along channel_zyx: 0=Z,1=Y,2=X.",
            "parent_region_id": crop["source_id"], "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
            "raw": {"path": r["raw_path"], "sha256": r["raw_sha256"], "shape_zyx": r["shape_zyx"], "dtype": r["dtype"]},
            "provenance": {"protocol": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-002", "sampler": DECLARED_RULE["sampler_id"], "contrast_class": crop["contrast_class"]},
            "diagnostic_only": True,
            "note": "AXIS-NEUTRAL DIAGNOSTIC. C/E-like high-contrast crops remain training-rejected; this experiment is not a training cohort.",
            "review_state": "UNREVIEWED", "schema_version": 1,
            "source_origin_xyz": [r["bounds_xyz"][a][0] for a in ("x", "y", "z")],
            "split": "G3_AXIS_NEUTRAL_DIAGNOSTIC", "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
        }
        (crop_dir / "workspace.json").write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # interleave (location, axis): location-major, axis interleaved
        questions = []
        axis_names = {0: "Z", 1: "Y", 2: "X"}
        qi = 0
        for li, loc in enumerate(locs, 1):
            for axis in (0, 1, 2):
                qi += 1
                left, right, _ = loc["edges"][axis]
                questions.append({
                    "id": f"MV-G3-AXNEU-L{li:02d}-{axis_names[axis]}",
                    "kind": "RAW_EM_INTERFACE_MEMBER",
                    "interface_id": f"MV-G3-AXNEU-L{li:02d}-{axis_names[axis]}",
                    "interface_member": 1, "interface_members": 1,
                    "location_index": li, "location_center_zyx": loc["center_zyx"],
                    "pair_left_zyx": list(left), "pair_right_zyx": list(right), "channel_zyx": axis,
                    "axis_name": axis_names[axis], "location_contrast_key": loc["contrast_key"],
                    "raw_gradient_score": loc["contrast_key"],
                    "selection": DECLARED_RULE["sampler_id"], "model_navigation": False,
                })
        args.queue_root.mkdir(parents=True, exist_ok=True)
        queue = {
            "schema_version": 1, "id": f"MV-G3-AXNEU-QUEUE-{crop['crop_id']}-006", "created_at": _now(),
            "status": "EXPERT_INTERFACE_REVIEW_REQUIRED", "workspace_id": ws_id, "crop_id": crop["crop_id"],
            "raw_sha256": r["raw_sha256"],
            "selection": {"method": DECLARED_RULE["sampler_id"], "declared_sampling_rule": DECLARED_RULE,
                          "contrast_class": crop["contrast_class"], "excluded_historical_edges": len(excluded),
                          "axis_counts": {"Z": len(locs), "Y": len(locs), "X": len(locs)}},
            "questions": questions,
            "scientific_boundary": "Axis-neutral paired-location diagnostic; each edge is an independent expert decision; no class targeted; not a training cohort.",
        }
        (args.queue_root / f"{crop['crop_id']}.json").write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
        selected.append({"crop_id": crop["crop_id"], "source_id": crop["source_id"], "contrast_class": crop["contrast_class"],
                         "workspace_id": ws_id, "locations": len(locs), "edges": 3 * len(locs),
                         "location_centers": [l["center_zyx"] for l in locs], "location_contrast": [round(l["contrast_key"], 3) for l in locs]})

    protocol = {
        "id": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-002", "schema_version": 1, "created_at": _now(),
        "status": "PREDECLARED_FIXED_BEFORE_LABELS", "supersedes": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-001 (too few locations, contrast-unbalanced; preserved, not overwritten)",
        "declared_sampling_rule": DECLARED_RULE, "manifest": {"path": str(MANIFEST.resolve()), "sha256": _sha(MANIFEST)},
        "excluded_historical_edges": len(excluded),
        "selected_crops": selected,
        "totals": {"crops": len(selected), "locations": sum(s["locations"] for s in selected), "edges": sum(s["edges"] for s in selected)},
    }
    PROTOCOL_OUT.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"totals": protocol["totals"], "selected": [(s["crop_id"], s["contrast_class"], s["locations"], s["location_contrast"]) for s in selected], "out": str(PROTOCOL_OUT.resolve())}, indent=2))


if __name__ == "__main__":
    main()
