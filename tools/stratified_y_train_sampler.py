"""Deterministic, label-blind stratified sampler over the Y-oriented TRAIN
interface candidate population.

Purpose: give the next human-review batch broad geometric/raw-EM coverage
across all six TRAIN crops and across raw-EM rank/score quantile bands, instead
of repeatedly sampling the top-gradient tail. Stratification is for COVERAGE
BREADTH ONLY. It does NOT encode any hypothesis about where DIFFERENT_PROCESS
interfaces occur, and it never consults expert labels, class predictions, or
prior SAME/DIFFERENT outcomes.

Declared allocation plan (fixed before any new label is seen):
  * bands over the ranked greedy-separated Y centre list (rank 0 == highest
    raw-EM gradient): 0-10%, 10-25%, 25-50%, 50-75%, 75-90%, 90-100%;
  * 6 TRAIN crops x 6 bands, 1 interface per (crop, band) => up to 36 interfaces;
  * within a band, pick the highest-ranked separated centre not already used in
    a prior queue and not conflicting (exact 12-voxel predicate) with centres
    already chosen in this batch.

Selection inputs allowed: raw-EM gradient geometry, Y orientation, TRAIN split,
crop identity, deterministic rank/quantile bands, exclusion of prior pairs.
Selection inputs forbidden: SAME/DIFFERENT labels, class predictions, affinity
model outputs, previously observed decisions.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Reuse the exact eligibility + separation semantics proven equivalent in tests.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_y_train_candidate_pool import (
    SEP2,
    eligible_y_population,
    greedy_separated_hashed,
)

REPO = Path(__file__).resolve().parents[1]

TRAIN = {
    "MV-G3-TRAIN-A": ("g3-external-review-packages-001/MV-G3-TRAIN-A/workspace.json",
                       ["g3-interface-queues-001/MV-G3-TRAIN-A.json", "g3-interface-queues-003/MV-G3-TRAIN-A.json"]),
    "MV-G3-TRAIN-B": ("g3-external-review-packages-001/MV-G3-TRAIN-B/workspace.json",
                       ["g3-interface-queues-001/MV-G3-TRAIN-B.json", "g3-interface-queues-003/MV-G3-TRAIN-B.json"]),
    "MV-G3-TRAIN-D": ("g3-external-review-packages-001/MV-G3-TRAIN-D/workspace.json",
                       ["g3-interface-queues-001/MV-G3-TRAIN-D.json", "g3-interface-queues-003/MV-G3-TRAIN-D.json"]),
    "MV-G3-TRAIN-F": ("g3-external-review-packages-001/MV-G3-TRAIN-F/workspace.json",
                       ["g3-interface-queues-001/MV-G3-TRAIN-F.json", "g3-interface-queues-003/MV-G3-TRAIN-F.json"]),
    "MV-G3-TRAIN-G": ("g3-external-review-packages-002/MV-G3-TRAIN-G/workspace.json",
                       ["g3-interface-queues-002/MV-G3-TRAIN-G.json", "g3-interface-queues-003/MV-G3-TRAIN-G.json"]),
    "MV-G3-TRAIN-H": ("g3-external-review-packages-002/MV-G3-TRAIN-H/workspace.json",
                       ["g3-interface-queues-002/MV-G3-TRAIN-H.json", "g3-interface-queues-003/MV-G3-TRAIN-H.json"]),
}
BANDS = ((0.0, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.90), (0.90, 1.0))
BAND_LABELS = ["0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"]
MARGIN = 3

# Declared verbatim BEFORE any new label is recorded. Embedded immutably in
# every artifact so an auditor can distinguish the sampling rule fixed now from
# the SAME/DIFFERENT labels obtained afterward.
DECLARED_SAMPLING_RULE = {
    "sampler_id": "STRATIFIED_Y_TRAIN_RANK_BANDS_V1",
    "declared_before_labels": True,
    "population": "interior voxels whose argmax(|3D raw-EM gradient|) axis == Y(channel 1)",
    "candidate_ordering": "descending raw-EM gradient magnitude score (stable sort); rank 0 == highest score",
    "selectable_centres": "greedy 12-voxel spatially-separated centres over the ranked eligible population (exact predicate d^2 >= 144)",
    "band_boundaries_fraction_of_separated_pool": [
        {"label": "0-10%", "lo": 0.0, "hi": 0.10},
        {"label": "10-25%", "lo": 0.10, "hi": 0.25},
        {"label": "25-50%", "lo": 0.25, "hi": 0.50},
        {"label": "50-75%", "lo": 0.50, "hi": 0.75},
        {"label": "75-90%", "lo": 0.75, "hi": 0.90},
        {"label": "90-100%", "lo": 0.90, "hi": 1.0},
    ],
    "band_index_math": "start = floor(lo * n_separated); end = ceil(hi * n_separated); ranks [start, end)",
    "allocation": "1 interface per (crop, band) by default (per_band configurable); 6 TRAIN crops x 6 bands",
    "within_band_pick": "lowest rank (highest gradient) in band that is interior, in-bounds, not a prior exact pair, and 12-voxel separated from centres already chosen in this batch",
    "interface_geometry": "boundary: center member offset (0,0) plus neighbours (-1,0),(1,0) along the two axes orthogonal to Y; positive neighbour along Y is the affinity pair",
    "allowed_selection_inputs": ["raw-EM 3D gradient", "Y orientation", "TRAIN split", "crop identity", "deterministic rank/quantile bands", "exclusion of prior exact pairs"],
    "forbidden_selection_inputs": ["SAME/DIFFERENT labels", "target-class predictions", "affinity model outputs", "previously observed SAME/DIFFERENT decisions"],
    "purpose": "broad geometric/raw-EM coverage; NOT targeting the missing DIFFERENT_PROCESS class",
}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def prior_pairs(queue_paths: list[Path]) -> set:
    pairs: set = set()
    for qp in queue_paths:
        if not qp.exists():
            continue
        for q in json.loads(qp.read_text())["questions"]:
            pairs.add((tuple(q["pair_left_zyx"]), tuple(q["pair_right_zyx"]), int(q["channel_zyx"])))
    return pairs


def _member_pairs(centre: tuple, arr_shape: tuple) -> list[tuple]:
    """Reproduce the generator's boundary interface geometry for axis Y (1):
    center member + two neighbours offset along the two axes orthogonal to Y."""
    axis = 1
    z, y, x = centre
    other = [d for d in range(3) if d != axis]
    offsets = ((0, 0), (-1, 0), (1, 0))
    members = []
    for off in offsets:
        left = [z, y, x]
        left[other[0]] += off[0]
        left[other[1]] += off[1]
        right = left[:axis] + [left[axis] + 1] + left[axis + 1:]
        members.append((tuple(left), tuple(right), axis))
    return members


def _interior(centre: tuple, arr_shape: tuple) -> bool:
    return all(MARGIN <= c < s - MARGIN for c, s in zip(centre, arr_shape))


def select_for_crop(arr: np.ndarray, excluded: set, per_band: int) -> list[dict]:
    """Deterministic band-stratified selection for one crop. Returns interface
    records with band label + rank. Never reads labels."""
    coords, scores = eligible_y_population(arr)
    score_by_voxel = {(int(c[0]), int(c[1]), int(c[2])): float(s) for c, s in zip(coords, scores)}
    separated = greedy_separated_hashed(coords)  # list[(z,y,x)] in rank order
    n = len(separated)
    chosen: list[dict] = []
    chosen_centres: list[tuple] = []
    for (lo, hi), label in zip(BANDS, BAND_LABELS):
        start = int(np.floor(lo * n))
        end = int(np.ceil(hi * n))
        picked_in_band = 0
        for rank in range(start, min(end, n)):
            if picked_in_band >= per_band:
                break
            centre = separated[rank]
            if not _interior(centre, arr.shape):
                continue
            members = _member_pairs(centre, arr.shape)
            # exclude prior exact pairs
            if any(m in excluded for m in members):
                continue
            # ensure every member pair is in-bounds
            if any(any(v < 0 or v >= s for v, s in zip(m[0], arr.shape)) or m[1][1] >= arr.shape[1] for m in members):
                continue
            # exact separation against centres already chosen this batch
            if any((centre[0] - c[0]) ** 2 + (centre[1] - c[1]) ** 2 + (centre[2] - c[2]) ** 2 < SEP2 for c in chosen_centres):
                continue
            chosen.append({"centre_zyx": list(centre), "band": label, "rank_within_separated": rank, "separated_pool_size": n, "members": members, "raw_gradient_score": score_by_voxel[(centre[0], centre[1], centre[2])]})
            chosen_centres.append(centre)
            picked_in_band += 1
    return chosen


def build(workspace_out: Path, queue_out: Path, per_band: int) -> dict:
    if workspace_out.exists() or queue_out.exists():
        raise FileExistsError("Refusing to overwrite existing -004 workspace/queue")
    summary = []
    total_interfaces = 0
    total_questions = 0
    for crop_id, (ws_rel, prior_rels) in TRAIN.items():
        source = json.loads((REPO / "experiments/phase6e" / ws_rel).read_text())
        arr = np.asarray(np.load(source["raw"]["path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        excluded = prior_pairs([REPO / "experiments/phase6e" / r for r in prior_rels])
        picks = select_for_crop(arr, excluded, per_band)

        crop_dir = workspace_out / crop_id
        crop_dir.mkdir(parents=True)
        event_log = crop_dir / "workspace.events.jsonl"
        event_log.write_text("", encoding="utf-8")
        ws_id = f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id[-1]}-004"
        workspace = {
            "allowed_decisions": ["BAD_QUESTION", "DIFFERENT_PROCESS", "SAME_PROCESS", "UNCERTAIN"],
            "coordinate_frame": source.get("coordinate_frame"),
            "created_at": _now(),
            "crop_id": crop_id,
            "event_log": {"append_only": True, "path": str(event_log.resolve())},
            "id": ws_id,
            "pair_contract": source.get("pair_contract"),
            "parent_region_id": source.get("parent_region_id"),
            "prohibited_promotions": source.get("prohibited_promotions"),
            "raw": source["raw"],
            "provenance": {
                "derived_from_workspace": {"path": str((REPO / "experiments/phase6e" / ws_rel).resolve()), "id": source["id"]},
                "sampler": "STRATIFIED_Y_TRAIN_RANK_BANDS_V1",
                "reason": "Broad label-blind Y/TRAIN coverage across raw-EM rank bands and crops",
            },
            "review_state": "UNREVIEWED",
            "schema_version": 1,
            "source_origin_xyz": source.get("source_origin_xyz"),
            "split": "G3_TARGET_TRAIN",
            "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
        }
        (crop_dir / "workspace.json").write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        questions = []
        for i, pick in enumerate(picks, 1):
            for member_index, (left, right, axis) in enumerate(pick["members"], 1):
                questions.append({
                    "id": f"MV-G3-IF-{i:03d}-{member_index:02d}",
                    "kind": "RAW_EM_INTERFACE_MEMBER",
                    "interface_id": f"MV-G3-IF-{i:03d}",
                    "interface_member": member_index,
                    "interface_members": 3,
                    "pair_left_zyx": list(left),
                    "pair_right_zyx": list(right),
                    "channel_zyx": axis,
                    "band": pick["band"],
                    "rank_within_separated": pick["rank_within_separated"],
                    "raw_gradient_score": pick["raw_gradient_score"],
                    "selection": "STRATIFIED_Y_TRAIN_RANK_BANDS_V1",
                    "model_navigation": False,
                })
        queue = {
            "schema_version": 1,
            "id": f"MV-G3-RAW-Y-STRATIFIED-QUEUE-{crop_id[-1]}-004",
            "created_at": _now(),
            "status": "EXPERT_INTERFACE_REVIEW_REQUIRED",
            "workspace_id": ws_id,
            "crop_id": crop_id,
            "raw_sha256": source["raw"]["sha256"],
            "selection": {
                "method": "STRATIFIED_Y_TRAIN_RANK_BANDS_V1",
                "bands": BAND_LABELS,
                "interfaces_per_band": per_band,
                "orientation_axis": "Y",
                "orientation_axis_channel_zyx": 1,
                "label_blind": True,
                "prohibited_inputs": ["SegNeuron", "supervoxels", "prior labels", "reviewed SAME/DIFFERENT decisions", "affinity model predictions"],
                "band_allocation": {p["band"]: sum(1 for x in picks if x["band"] == p["band"]) for p in picks} if picks else {},
                "excluded_prior_pairs": len(excluded),
                "declared_sampling_rule": DECLARED_SAMPLING_RULE,
            },
            "questions": questions,
            "scientific_boundary": "Raw-EM rank-band stratification chooses review locations for coverage only. Each direct pair remains an independent expert decision; stratification does not target any biological class.",
        }
        queue_dir = queue_out
        queue_dir.mkdir(parents=True, exist_ok=True)
        (queue_dir / f"{crop_id}.json").write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")

        interfaces = len(picks)
        total_interfaces += interfaces
        total_questions += len(questions)
        band_counts = {label: sum(1 for x in picks if x["band"] == label) for label in BAND_LABELS}
        summary.append({"crop_id": crop_id, "workspace_id": ws_id, "interfaces": interfaces, "questions": len(questions), "band_counts": band_counts})
    manifest = {
        "id": "MV-G3-Y-TRAIN-STRATIFIED-REVIEW-BATCH-004",
        "created_at": _now(),
        "status": "EXPERT_INTERFACE_REVIEW_REQUIRED",
        "declared_sampling_rule": DECLARED_SAMPLING_RULE,
        "per_band_target": per_band,
        "crops": list(TRAIN),
        "totals": {"interfaces": total_interfaces, "questions": total_questions},
        "per_crop": summary,
        "note": "Declared sampling rule fixed at created_at, before any -004 SAME/DIFFERENT label exists.",
    }
    (queue_out / "batch-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"total_interfaces": total_interfaces, "total_questions": total_questions, "per_crop": summary, "manifest": str((queue_out / "batch-manifest.json").resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-band", type=int, default=1, help="interfaces per (crop, band) cell")
    parser.add_argument("--workspace-root", type=Path, default=REPO / "experiments/phase6e/g3-external-review-packages-004")
    parser.add_argument("--queue-root", type=Path, default=REPO / "experiments/phase6e/g3-interface-queues-004")
    args = parser.parse_args()
    result = build(args.workspace_root, args.queue_root, args.per_band)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
