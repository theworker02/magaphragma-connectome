"""Freeze a cross-region G1/ELF diagnostic matrix before G2 review or training."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tifffile


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stats(values: np.ndarray) -> dict[str, float]:
    return {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std()),
            "p01": float(np.percentile(values, 1)), "p50": float(np.percentile(values, 50)), "p99": float(np.percentile(values, 99)),
            "fraction_ge_0_9": float((values >= 0.9).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="phase6d output root")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen diagnostic matrix: {args.output}")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    rows = []
    for region in cohort["regions"]:
        identifier = region["id"]
        prediction = args.root / f"g1-diagnostic-{identifier}-step2280"
        true3d = args.root / f"g1-diagnostic-{identifier}-step2280-true3d-001"
        inference = json.loads((prediction / "inference.json").read_text(encoding="utf-8"))
        queue = json.loads((prediction / "g2-diagnostic-queue.json").read_text(encoding="utf-8"))
        stage = json.loads((true3d / "receipt.json").read_text(encoding="utf-8"))
        edge = json.loads((true3d / "edge-evidence.json").read_text(encoding="utf-8"))
        if inference["input"]["sha256"] != region["raw_sha256"] or inference["checkpoint"]["sha256"] != args.checkpoint_sha256:
            raise ValueError(f"{identifier}: output does not match frozen raw input/checkpoint")
        affinity = np.load(prediction / "affinities.npy", allow_pickle=False)
        boundary = tifffile.imread(prediction / "boundaries.tif")
        labels = np.load(true3d / "labels.npy", allow_pickle=False)
        _, sizes = np.unique(labels, return_counts=True)
        table = np.load(true3d / "edge-evidence.npz", allow_pickle=False)
        split = table["split_probability"]
        rows.append({
            "region_id": identifier, "role": region["role"], "bounds_xyz": region["bounds_xyz"],
            "raw_sha256": region["raw_sha256"], "inference": {"receipt_sha256": digest(prediction / "inference.json"), "affinity": stats(affinity), "boundary": stats(boundary)},
            "true3d": {"receipt_sha256": digest(true3d / "receipt.json"), "watershed": stage["watershed"], "fragments": stage["fragments"], "rag": {"nodes": stage["rag"]["nodes"], "edges": stage["rag"]["edges"]},
                       "split_probability": {"min": float(split.min()), "median": float(np.median(split)), "max": float(split.max()), "edges_gt_0_5": int((split > .5).sum()), "edges_gt_0_75": int((split > .75).sum()), "edges_gt_0_9": int((split > .9).sum())},
                       "costs": {"negative": int((table["transformed_cost"] < 0).sum()), "positive": int((table["transformed_cost"] > 0).sum())},
                       "instances": {"count": int(sizes.size), "largest_fraction": float(sizes.max() / labels.size), "size_min": int(sizes.min()), "size_median": float(np.median(sizes)), "size_max": int(sizes.max())}},
            "raw_em_correspondence": {"status": "EXPERT_REVIEW_REQUIRED", "queue": str((prediction / "g2-diagnostic-queue.json").resolve()), "queue_sha256": digest(prediction / "g2-diagnostic-queue.json"), "merge_confident_raw_boundary_candidates": queue["candidate_counts_before_nms"]["merge_confident_raw_boundary"], "continuity_controls": queue["candidate_counts_before_nms"]["continuity_control"], "frozen_questions": len(queue["questions"])},
            "scientific_boundary": "Machine diagnostics only; no question is a biological label until an identified expert reviews it."
        })
    document = {"schema_version": 1, "id": "MV-G2-FROZEN-G1-DIAGNOSTIC-MATRIX-001", "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "status": "FROZEN_BEFORE_EXPERT_REVIEW_AND_G2_TRAINING", "cohort": {"path": str(args.cohort.resolve()), "sha256": digest(args.cohort)}, "checkpoint_sha256": args.checkpoint_sha256,
                "regions": rows, "promotion_rule": "No G2 training until expert-reviewed SAME and DIFFERENT decisions exist in every required TRAIN and VALIDATION region; MV-GTVOL-000004 remains excluded.",
                "prohibited_actions": ["G2 training", "checkpoint selection", "MV-GTVOL-000004 replay", "MV-FRAG", "MV-N", "MV-CONN"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"id": document["id"], "regions": len(rows), "status": document["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
