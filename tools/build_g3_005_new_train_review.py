"""Provision -005 review packages for the newly selected G3 TRAIN crops (I-L).

Creates fresh immutable workspaces (new IDs, empty append-only logs) bound to
the new raw crops, and a label-blind interface queue per crop stratified across
ALL THREE axes (Z/Y/X) and raw-EM rank bands, with spatially-separated centres.

Sampling is predeclared and label-blind: it uses raw-EM gradient geometry,
orientation axis, and rank bands only. It never reads SAME/DIFFERENT labels or
any prior crop's class composition. Meaningful axis coverage (not just Y) is a
direct response to the identifiability contract's need for within-region
SAME/DIFFERENT contrast on some axis; which axis/class each interface turns out
to be is decided ONLY by the expert.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "experiments/phase6e/G3_NEW_REGION_SELECTION_PROTOCOL_001.json"
MARGIN = 3
SEP2 = 12 ** 2
BANDS = ((0.0, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.90), (0.90, 1.0))
BAND_LABELS = ["0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"]
AXES = {0: "Z", 1: "Y", 2: "X"}

DECLARED_SAMPLING_RULE = {
    "sampler_id": "STRATIFIED_AXIS_x_RANKBAND_V1",
    "declared_before_labels": True,
    "population": "interior voxels; each candidate assigned to its dominant |3D raw-EM gradient| axis (Z/Y/X)",
    "candidate_ordering": "descending raw-EM gradient magnitude within each axis; rank 0 == highest",
    "selectable_centres": "greedy 12-voxel spatially-separated centres per axis (exact d^2>=144)",
    "stratification": "for each axis independently: 6 rank bands 0-10..90-100%; 1 interface per (axis, band) => up to 18 interfaces/crop",
    "within_band_pick": "lowest rank in band that is interior, in-bounds, and 12-voxel separated from centres already chosen this crop (across all axes)",
    "interface_geometry": "center member + neighbours (-1,0),(1,0) along the two axes orthogonal to the affinity axis; positive neighbour along the affinity axis is the pair",
    "allowed_selection_inputs": ["raw-EM 3D gradient", "dominant-axis orientation", "rank/quantile bands", "crop identity"],
    "forbidden_selection_inputs": ["SAME/DIFFERENT labels", "class predictions", "learned affinities", "prior crop class composition"],
    "purpose": "broad axis + raw-EM coverage to give a genuine chance of within-region SAME/DIFFERENT contrast; NOT targeting any class",
}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def eligible_axis_population(arr: np.ndarray):
    grads = np.stack(np.gradient(arr))
    score = np.linalg.norm(grads, axis=0)
    dominant = np.argmax(np.abs(grads), axis=0)
    interior = np.zeros(arr.shape, dtype=bool)
    interior[MARGIN:arr.shape[0]-MARGIN, MARGIN:arr.shape[1]-MARGIN, MARGIN:arr.shape[2]-MARGIN] = True
    out = {}
    for axis in (0, 1, 2):
        mask = interior & (dominant == axis)
        coords = np.argwhere(mask)
        vals = score[mask]
        order = np.argsort(vals, kind="stable")[::-1]
        out[axis] = (coords[order], vals[order])
    return out


def greedy_separated(coords):
    sep = 12
    grid = {}
    kept = []
    for zyx in coords:
        z, y, x = int(zyx[0]), int(zyx[1]), int(zyx[2])
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
            kept.append((z, y, x))
    return kept


def _members(centre, axis):
    z, y, x = centre
    other = [d for d in range(3) if d != axis]
    out = []
    for off in ((0, 0), (-1, 0), (1, 0)):
        left = [z, y, x]
        left[other[0]] += off[0]
        left[other[1]] += off[1]
        right = left[:axis] + [left[axis] + 1] + left[axis+1:]
        out.append((tuple(left), tuple(right)))
    return out


def select_crop(arr, score_lookup, pops):
    chosen = []
    chosen_centres = []
    for axis in (0, 1, 2):
        coords, vals = pops[axis]
        sep = greedy_separated(coords)
        n = len(sep)
        score_at = {}
        # build score lookup for this axis's separated centres
        val_by_voxel = {(int(c[0]), int(c[1]), int(c[2])): float(v) for c, v in zip(coords, vals)}
        for (lo, hi), label in zip(BANDS, BAND_LABELS):
            start, end = int(np.floor(lo*n)), int(np.ceil(hi*n))
            picked = False
            for rank in range(start, min(end, n)):
                if picked:
                    break
                centre = sep[rank]
                if not all(MARGIN <= c < s-MARGIN for c, s in zip(centre, arr.shape)):
                    continue
                if any((centre[0]-c[0])**2+(centre[1]-c[1])**2+(centre[2]-c[2])**2 < SEP2 for c in chosen_centres):
                    continue
                chosen.append({"centre": centre, "axis": axis, "band": label, "rank": rank, "pool": n,
                               "score": val_by_voxel[(centre[0], centre[1], centre[2])]})
                chosen_centres.append(centre)
                picked = True
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=REPO/"experiments/phase6e/g3-external-review-packages-005")
    parser.add_argument("--queue-root", type=Path, default=REPO/"experiments/phase6e/g3-interface-queues-005")
    args = parser.parse_args()
    if args.workspace_root.exists() or args.queue_root.exists():
        raise SystemExit("Refusing to overwrite existing -005 artifacts")
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads((REPO/"local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json").read_text())
    src_rec = {r["id"]: r for r in manifest["records"]}

    summary = []
    for sel in protocol["selected"]:
        crop_id = sel["new_crop_id"]
        rec = src_rec[sel["source_id"]]
        raw_path = rec["raw_path"]
        arr = np.asarray(np.load(raw_path, mmap_mode="r", allow_pickle=False), dtype=np.float32)
        pops = eligible_axis_population(arr)
        picks = select_crop(arr, None, pops)

        crop_dir = args.workspace_root / crop_id
        crop_dir.mkdir(parents=True)
        log = crop_dir / "workspace.events.jsonl"
        log.write_text("", encoding="utf-8")
        ws_id = f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id[-1]}-005"
        workspace = {
            "allowed_decisions": ["BAD_QUESTION", "DIFFERENT_PROCESS", "SAME_PROCESS", "UNCERTAIN"],
            "coordinate_frame": "MV-FRAME-DVID-WASP5-001",
            "created_at": _now(),
            "crop_id": crop_id,
            "event_log": {"append_only": True, "path": str(log.resolve())},
            "id": ws_id,
            "pair_contract": "pair_left_zyx is followed by its positive neighbour along channel_zyx: 0=Z, 1=Y, 2=X.",
            "parent_region_id": sel["source_id"],
            "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
            "raw": {"path": raw_path, "sha256": sel["raw_sha256"], "shape_zyx": rec["shape_zyx"], "dtype": rec["dtype"]},
            "provenance": {"selection_protocol": {"path": str(PROTOCOL.resolve()), "id": protocol["id"]}, "sampler": DECLARED_SAMPLING_RULE["sampler_id"]},
            "review_state": "UNREVIEWED",
            "schema_version": 1,
            "source_origin_xyz": [rec["bounds_xyz"][a][0] for a in ("x", "y", "z")],
            "split": "G3_TARGET_TRAIN",
            "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
        }
        (crop_dir / "workspace.json").write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        questions = []
        for i, p in enumerate(picks, 1):
            axis = p["axis"]
            for m, (left, right) in enumerate(_members(p["centre"], axis), 1):
                questions.append({
                    "id": f"MV-G3-IF-{i:03d}-{m:02d}", "kind": "RAW_EM_INTERFACE_MEMBER",
                    "interface_id": f"MV-G3-IF-{i:03d}", "interface_member": m, "interface_members": 3,
                    "pair_left_zyx": list(left), "pair_right_zyx": list(right), "channel_zyx": axis,
                    "axis_name": AXES[axis], "band": p["band"], "rank_within_separated": p["rank"],
                    "raw_gradient_score": p["score"], "selection": "STRATIFIED_AXIS_x_RANKBAND_V1", "model_navigation": False,
                })
        args.queue_root.mkdir(parents=True, exist_ok=True)
        queue = {
            "schema_version": 1, "id": f"MV-G3-RAW-AXISBAND-QUEUE-{crop_id[-1]}-005", "created_at": _now(),
            "status": "EXPERT_INTERFACE_REVIEW_REQUIRED", "workspace_id": ws_id, "crop_id": crop_id,
            "raw_sha256": sel["raw_sha256"],
            "selection": {"method": "STRATIFIED_AXIS_x_RANKBAND_V1", "declared_sampling_rule": DECLARED_SAMPLING_RULE,
                          "axis_band_counts": {AXES[a]: {b: sum(1 for p in picks if p["axis"] == a and p["band"] == b) for b in BAND_LABELS} for a in (0, 1, 2)}},
            "questions": questions,
            "scientific_boundary": "Raw-EM axis x rank-band stratification chooses review locations for coverage only; each pair is an independent expert decision and no class is targeted.",
        }
        (args.queue_root / f"{crop_id}.json").write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
        axis_counts = {AXES[a]: sum(1 for p in picks if p["axis"] == a) for a in (0, 1, 2)}
        summary.append({"crop_id": crop_id, "source_id": sel["source_id"], "workspace_id": ws_id,
                        "interfaces": len(picks), "questions": len(questions), "axis_counts": axis_counts})

    manifest_out = {"id": "MV-G3-NEW-TRAIN-REVIEW-BATCH-005", "created_at": _now(),
                    "declared_sampling_rule": DECLARED_SAMPLING_RULE, "selection_protocol": protocol["id"], "per_crop": summary}
    (args.queue_root / "batch-manifest.json").write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"per_crop": summary}, indent=2))


if __name__ == "__main__":
    main()
