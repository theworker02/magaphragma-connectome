"""SeamSmith matcher ? candidate pair scoring; NEVER auto-merges.

Deterministic scoring formulas
------------------------------
spatial_distance:
  Euclidean distance between world centroids of face objects A and B.

direction_agreement:
  abs(dot(major_axis_A, major_axis_B))  # both unit; in [0,1]

morphology_agreement:
  0.5 * volume_ratio + 0.5 * bbox_iou
  where volume_ratio = min(Va,Vb) / max(Va,Vb)
  bbox_iou = intersection_volume / union_volume of axis-aligned bboxes (world)

overlap:
  |voxels_A intersect voxels_B| / min(|A_halo|, |B_halo|) in shared world halo coords

confidence:
  0.30 * overlap
  + 0.25 * direction_agreement
  + 0.25 * morphology_agreement
  + 0.20 * spatial_score
  where spatial_score = exp(-spatial_distance / (halo * 2))

recommended_action (thresholds, never auto-merge):
  MATCH  if confidence >= 0.75 and overlap >= 0.20 and morphology_agreement >= 0.50
  MERGE  if confidence >= 0.60 and overlap >= 0.35 and volume_ratio < 0.40
         (advisory only ? SeamSmith never applies merges)
  SPLIT  if overlap < 0.05 and spatial_distance < halo and direction_agreement < 0.30
  HOLD   if confidence < 0.35
  REVIEW otherwise
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from seamsmith.seams import FaceObject, SeamPair, SeamReport, extract_face_objects


def _bbox_iou_world(
    a: FaceObject,
    b: FaceObject,
    origin_a: tuple[int, int, int],
    origin_b: tuple[int, int, int],
) -> float:
    def world_bbox(obj: FaceObject, origin: tuple[int, int, int]):
        return [
            (obj.bbox_zyx[i][0] + origin[i], obj.bbox_zyx[i][1] + origin[i]) for i in range(3)
        ]

    wa = world_bbox(a, origin_a)
    wb = world_bbox(b, origin_b)
    inter_vol = 1
    vol_a = 1
    vol_b = 1
    for i in range(3):
        a0, a1 = wa[i]
        b0, b1 = wb[i]
        ia0, ia1 = max(a0, b0), min(a1, b1)
        len_i = max(0, ia1 - ia0 + 1)
        len_a = max(1, a1 - a0 + 1)
        len_b = max(1, b1 - b0 + 1)
        inter_vol *= len_i
        vol_a *= len_a
        vol_b *= len_b
    union = vol_a + vol_b - inter_vol
    if union <= 0:
        return 0.0
    return float(inter_vol / union)


def _world_voxel_keys(obj: FaceObject, origin: tuple[int, int, int]) -> set[tuple[int, int, int]]:
    keys: set[tuple[int, int, int]] = set()
    for z, y, x in obj.voxel_coords:
        keys.add((int(z) + origin[0], int(y) + origin[1], int(x) + origin[2]))
    return keys


def score_pair(
    a: FaceObject,
    b: FaceObject,
    *,
    origin_a: tuple[int, int, int],
    origin_b: tuple[int, int, int],
    halo: int,
) -> dict[str, Any]:
    ca = np.asarray(a.centroid_zyx, dtype=np.float64)
    cb = np.asarray(b.centroid_zyx, dtype=np.float64)
    spatial_distance = float(np.linalg.norm(ca - cb))

    ax = np.asarray(a.major_axis, dtype=np.float64)
    bx = np.asarray(b.major_axis, dtype=np.float64)
    direction_agreement = float(abs(np.dot(ax, bx)))

    va, vb = max(1, a.volume), max(1, b.volume)
    volume_ratio = float(min(va, vb) / max(va, vb))
    bbox_iou = _bbox_iou_world(a, b, origin_a, origin_b)
    morphology_agreement = float(0.5 * volume_ratio + 0.5 * bbox_iou)

    ka = _world_voxel_keys(a, origin_a)
    kb = _world_voxel_keys(b, origin_b)
    inter = len(ka & kb)
    denom = max(1, min(len(ka), len(kb)))
    overlap = float(inter / denom)

    spatial_score = float(math.exp(-spatial_distance / max(1.0, 2.0 * halo)))
    confidence = float(
        0.30 * overlap
        + 0.25 * direction_agreement
        + 0.25 * morphology_agreement
        + 0.20 * spatial_score
    )
    confidence = max(0.0, min(1.0, confidence))

    if confidence >= 0.75 and overlap >= 0.20 and morphology_agreement >= 0.50:
        action = "MATCH"
    elif confidence >= 0.60 and overlap >= 0.35 and volume_ratio < 0.40:
        action = "MERGE"
    elif overlap < 0.05 and spatial_distance < halo and direction_agreement < 0.30:
        action = "SPLIT"
    elif confidence < 0.35:
        action = "HOLD"
    else:
        action = "REVIEW"

    evidence = {
        "spatial_distance": round(spatial_distance, 6),
        "direction_agreement": round(direction_agreement, 6),
        "morphology_agreement": round(morphology_agreement, 6),
        "volume_ratio": round(volume_ratio, 6),
        "bbox_iou": round(bbox_iou, 6),
        "overlap": round(overlap, 6),
        "overlap_voxels": inter,
        "spatial_score": round(spatial_score, 6),
        "confidence": round(confidence, 6),
    }
    return {
        "a": {"chunk": a.chunk, "label_id": a.label_id, "face": a.face, "volume": a.volume},
        "b": {"chunk": b.chunk, "label_id": b.label_id, "face": b.face, "volume": b.volume},
        "evidence": evidence,
        "recommended_action": action,
        "auto_merge": False,
    }


def match_seam(seam: SeamPair) -> SeamReport:
    """Extract face objects and score all A x B candidates. Never merges labels."""
    objs_a = extract_face_objects(seam.chunk_a, seam.a_face, seam.halo)
    objs_b = extract_face_objects(seam.chunk_b, seam.b_face, seam.halo)
    candidates: list[dict[str, Any]] = []
    for a in objs_a:
        for b in objs_b:
            candidates.append(
                score_pair(
                    a,
                    b,
                    origin_a=seam.chunk_a.origin_zyx,
                    origin_b=seam.chunk_b.origin_zyx,
                    halo=seam.halo,
                )
            )
    candidates.sort(
        key=lambda c: (
            -c["evidence"]["confidence"],
            c["a"]["label_id"],
            c["b"]["label_id"],
        )
    )
    return SeamReport(
        candidates=candidates,
        meta={
            "axis": seam.axis,
            "halo": seam.halo,
            "n_objects_a": len(objs_a),
            "n_objects_b": len(objs_b),
            "auto_merge_applied": False,
            "policy": "NEVER_AUTO_MERGE",
        },
    )
