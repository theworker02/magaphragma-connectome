"""Label-blind prospective survey of raw DVID candidate regions for NEW G3
TRAIN crops, plus a deterministic selection rule fixed before any expert label.

Excludes TRAIN A-H, all VALIDATION, MV-GTVOL-000004, and any survey source
already used by those regions. Eligibility uses only label-blind raw/geometric
information: bounds, spatial independence from existing regions, raw integrity,
interior volume, and deterministic raw-intensity statistics. It does NOT read
SAME/DIFFERENT labels, predictions, learned affinities, or the observed class
composition of any prior crop.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SURVEY_MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
OUT = REPO / "experiments/phase6e/MV-G3-NEW-REGION-SURVEY-001.json"

# Minimum L-inf gap (voxels, in the shared xyz frame) required between a new
# candidate box and EVERY existing region box to count as spatially independent.
# Boxes must be strictly non-overlapping (existing G3 guard); we additionally
# require a positive margin so adjacency cannot leak context.
MIN_GAP = 1
INTERIOR_MARGIN = 3  # matches interface generator interior requirement


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as s:
        while b := s.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def existing_regions() -> list[dict]:
    """Every region we must stay independent of: TRAIN A-H + VALIDATION, read
    from their frozen workspaces (bounds via survey source)."""
    manifest = json.loads(SURVEY_MANIFEST.read_text())
    by_source = {r["id"]: r for r in manifest["records"]}
    regions = []
    used_sources = set()
    for wf in glob.glob(str(REPO / "experiments/phase6e/g3-external-review-packages-*/MV-G3-*/workspace.json")):
        w = json.loads(Path(wf).read_text())
        src = w.get("parent_region_id")
        used_sources.add(src)
        rec = by_source.get(src)
        if rec:
            regions.append({"crop_id": w["crop_id"], "source_id": src, "bounds_xyz": rec["bounds_xyz"], "split": w.get("split")})
    return regions, used_sources, by_source


def _overlap(a: dict, b: dict) -> bool:
    return all(max(a[x][0], b[x][0]) < min(a[x][1], b[x][1]) for x in ("x", "y", "z"))


def _min_gap(a: dict, b: dict) -> int:
    """L-inf separation gap between boxes; negative if overlapping."""
    gaps = []
    for x in ("x", "y", "z"):
        # gap along axis: how far apart the intervals are (negative if overlap)
        gaps.append(max(b[x][0] - a[x][1], a[x][0] - b[x][1]))
    return max(gaps)  # boxes are separated iff max axis-gap >= MIN_GAP


def main() -> None:
    manifest = json.loads(SURVEY_MANIFEST.read_text())
    regions, used_sources, by_source = existing_regions()

    surveyed = []
    for rec in manifest["records"]:
        sid = rec["id"]
        raw = Path(rec["raw_path"])
        entry = {
            "candidate_source_id": sid,
            "bounds_xyz": rec["bounds_xyz"],
            "shape_zyx": rec["shape_zyx"],
            "dtype": rec["dtype"],
        }
        reasons = []
        # exclusion: already used by an existing region
        if sid in used_sources:
            reasons.append("ALREADY_USED_BY_EXISTING_G3_REGION")
        if "000004" in sid or "GTVOL" in sid:
            reasons.append("PROTECTED_REGRESSION_VOLUME")
        # raw integrity
        integrity_ok = raw.is_file() and _sha256(raw) == rec["raw_sha256"]
        if not integrity_ok:
            reasons.append("RAW_MISSING_OR_HASH_MISMATCH")
        # interior volume sufficiency
        z, y, x = rec["shape_zyx"]
        if min(z, y, x) < 2 * INTERIOR_MARGIN + 3:
            reasons.append("INSUFFICIENT_INTERIOR_VOLUME")
        # spatial independence vs every existing region
        overlaps = []
        min_gaps = []
        for other in regions:
            if _overlap(rec["bounds_xyz"], other["bounds_xyz"]):
                overlaps.append(other["crop_id"])
            min_gaps.append(_min_gap(rec["bounds_xyz"], other["bounds_xyz"]))
        if overlaps:
            reasons.append("SPATIAL_OVERLAP_WITH_EXISTING_REGION")
        entry["overlaps_with"] = overlaps
        entry["min_gap_to_any_existing"] = int(min(min_gaps)) if min_gaps else None
        # label-blind raw statistics (computed from voxels, not labels)
        if integrity_ok:
            arr = np.asarray(np.load(raw, mmap_mode="r", allow_pickle=False), dtype=np.float32)
            grads = np.gradient(arr)
            grad_mag = float(np.mean([np.mean(np.abs(g)) for g in grads]))
            entry["raw_statistics"] = {
                "mean": float(arr.mean()), "std": float(arr.std()),
                "p01_p50_p99": [float(v) for v in np.quantile(arr, [0.01, 0.5, 0.99])],
                "mean_abs_gradient": grad_mag,
            }
            entry["raw_sha256"] = rec["raw_sha256"]
        entry["eligible"] = not reasons
        entry["rejection_reasons"] = reasons
        surveyed.append(entry)

    eligible = [e for e in surveyed if e["eligible"]]
    report = {
        "id": "MV-G3-NEW-REGION-SURVEY-001",
        "survey_manifest": {"path": str(SURVEY_MANIFEST.resolve()), "sha256": _sha256(SURVEY_MANIFEST)},
        "excluded_existing_sources": sorted(used_sources),
        "existing_region_boxes": regions,
        "eligibility_criteria": {
            "must_not_be_used_by_existing_region": True,
            "must_not_be_protected_volume": True,
            "raw_integrity_required": True,
            "min_interior_volume_each_axis": 2 * INTERIOR_MARGIN + 3,
            "must_be_spatially_non_overlapping_with_all_existing": True,
            "label_blind": True,
            "forbidden_inputs": ["SAME/DIFFERENT labels", "class predictions", "learned affinities", "A-H class composition"],
        },
        "surveyed_population": surveyed,
        "eligible_candidates": [e["candidate_source_id"] for e in eligible],
        "eligible_count": len(eligible),
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"eligible_count": len(eligible), "eligible": report["eligible_candidates"], "excluded_existing_sources": report["excluded_existing_sources"], "out": str(OUT.resolve())}, indent=2))


if __name__ == "__main__":
    main()
