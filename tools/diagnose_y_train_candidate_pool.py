"""Label-blind characterization of the complete Y-oriented TRAIN candidate pool.

Reads raw EM only. It never opens any SAME/DIFFERENT decision and is not used
to select review candidates.

Design (per the required separation of concerns):
  1. Vectorized ELIGIBILITY over the whole volume:
       interior voxels whose dominant |gradient| axis == Y (channel 1).
     This is the complete geometrically-eligible Y population. No score cap,
     no separation, no truncation.
  2. RANKING/analysis of that reduced eligible set:
       extract eligible coords + scores, argsort scores descending, then
       report the full ranked population and its statistics.

The generator's 12-voxel spatial separation is a QUEUE-SELECTION rule, not an
eligibility property, so it is reported separately (how many separated centres
the greedy rule would yield) without shrinking the measured population.

Prior-interface centres are recovered by the generator's explicit convention:
the center member is the one with interface_member == 1 (offset (0,0)).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
MARGIN = 3
SEP2 = 12 ** 2
OUT = REPO / "experiments/phase6e/MV-G3-Y-TRAIN-CANDIDATE-POOL-DIAGNOSTIC.json"


def eligible_y_population(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Complete eligible Y population, ranked. No truncation.

    Returns (coords_zyx sorted by descending score, scores sorted descending).
    Eligibility = interior voxel AND dominant |gradient| axis is Y (channel 1).
    """
    grads = np.stack(np.gradient(arr))               # (3, Z, Y, X)
    score = np.linalg.norm(grads, axis=0)            # (Z, Y, X)
    dominant_axis = np.argmax(np.abs(grads), axis=0)  # (Z, Y, X)
    interior = np.zeros(arr.shape, dtype=bool)
    interior[MARGIN:arr.shape[0] - MARGIN, MARGIN:arr.shape[1] - MARGIN, MARGIN:arr.shape[2] - MARGIN] = True
    eligible = interior & (dominant_axis == 1)
    coords = np.argwhere(eligible)                   # (N, 3)
    values = score[eligible]                         # (N,)
    order = np.argsort(values, kind="stable")[::-1]
    return coords[order], values[order]


def greedy_separated_bruteforce(coords: np.ndarray) -> list[tuple]:
    """Reference greedy: walk ranked coords, accept if >= SEP from every
    accepted centre. O(N * kept). Semantics identical to the generator's
    `all((z-c0)**2+... >= SEP2 for c in centres)` acceptance predicate.
    """
    accepted: list[tuple] = []
    for zyx in coords:
        z, y, x = int(zyx[0]), int(zyx[1]), int(zyx[2])
        if all((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 >= SEP2 for c in accepted):
            accepted.append((z, y, x))
    return accepted


def greedy_separated_hashed(coords: np.ndarray) -> list[tuple]:
    """Spatial-hash greedy. Same ranking order, same acceptance predicate,
    same accepted centres in the same order as the brute-force reference; only
    the conflict SEARCH is accelerated via a grid whose cell size == SEP so any
    point within SEP must lie in the 27-cell neighbourhood.
    """
    sep = int(round(SEP2 ** 0.5))  # 12
    assert sep * sep == SEP2, "cell size must equal the exact separation radius"
    grid: dict[tuple, list[tuple]] = {}
    accepted: list[tuple] = []
    for zyx in coords:
        z, y, x = int(zyx[0]), int(zyx[1]), int(zyx[2])
        cell = (z // sep, y // sep, x // sep)
        conflict = False
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for (az, ay, ax) in grid.get((cell[0] + dz, cell[1] + dy, cell[2] + dx), ()):
                        if (az - z) ** 2 + (ay - y) ** 2 + (ax - x) ** 2 < SEP2:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            if conflict:
                break
        if not conflict:
            grid.setdefault(cell, []).append((z, y, x))
            accepted.append((z, y, x))
    return accepted


def greedy_separated_count(coords: np.ndarray) -> int:
    return len(greedy_separated_hashed(coords))


def prior_centres(queue_paths: list[Path]) -> set:
    """Distinct interface centres already asked, using the generator's explicit
    convention: the center member is interface_member == 1 (offset (0,0))."""
    centres: set[tuple] = set()
    for qp in queue_paths:
        if not qp.exists():
            continue
        queue = json.loads(qp.read_text())
        for q in queue["questions"]:
            if int(q.get("interface_member", -1)) == 1:
                centres.add(tuple(int(v) for v in q["pair_left_zyx"]))
    return centres


def rank_of(coords: np.ndarray, point: tuple) -> int | None:
    matches = np.argwhere((coords[:, 0] == point[0]) & (coords[:, 1] == point[1]) & (coords[:, 2] == point[2]))
    return int(matches[0, 0]) if len(matches) else None


def main() -> None:
    report = []
    for crop_id, (ws_rel, prior_rels) in TRAIN.items():
        ws = json.loads((REPO / "experiments/phase6e" / ws_rel).read_text())
        arr = np.asarray(np.load(ws["raw"]["path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        coords, scores = eligible_y_population(arr)
        n = int(scores.shape[0])
        priors = prior_centres([REPO / "experiments/phase6e" / r for r in prior_rels])
        prior_ranks = sorted(r for r in (rank_of(coords, p) for p in priors) if r is not None)
        q = np.quantile(scores, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist() if n else []
        # Concentration: fraction of eligible mass within +/-10% of the median score.
        concentration = None
        if n:
            med = float(np.median(scores))
            band = float(np.mean(np.abs(scores - med) <= 0.1 * med))
            concentration = round(band, 3)
        report.append({
            "crop_id": crop_id,
            "total_eligible_Y_voxels": n,
            "separated_centres_full_pool": greedy_separated_count(coords),
            "prior_selected_distinct_centres": len(priors),
            "prior_selections_found_in_eligible_pool": len(prior_ranks),
            "remaining_eligible_after_prior": n - len(prior_ranks),
            "score_quantiles_min_q25_med_q75_q90_max": [round(v, 3) for v in q],
            "prior_selection_ranks_within_eligible": prior_ranks,
            "deepest_prior_rank": max(prior_ranks) if prior_ranks else None,
            "eligible_depth_beyond_deepest_prior": (n - 1 - max(prior_ranks)) if prior_ranks else n,
            "fraction_within_10pct_of_median_score": concentration,
        })
    result = {
        "id": "MV-G3-Y-TRAIN-CANDIDATE-POOL-DIAGNOSTIC",
        "eligibility_definition": "interior voxel AND argmax(|gradient|) axis == Y(1); no score cap, no separation, no truncation",
        "separation_note": "12-voxel separation is a queue-SELECTION rule; reported as separated_centres_full_pool, not as eligibility",
        "label_blind": True,
        "per_crop": report,
        "cross_crop_prior_distribution": {r["crop_id"]: r["prior_selected_distinct_centres"] for r in report},
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
