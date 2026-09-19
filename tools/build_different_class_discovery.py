"""DIFFERENT-class discovery under equivariant UI (not AS2 revalidation, not -009).

Root cause of AS2 all-SAME: relative Q1–Q4 selection forced absolute mean face-step
into ~0.3–3.0 while historical C/E magnitudes are ~8–17+. This batch selects
single faces with absolute |ΔI| ≥ ABS_MIN on tissue that actually has C/E-scale
contrast (SURVEY-035 primary; unused 045–048 secondary), axis-balanced when
possible, spatially separated from the AS2-reviewed center on 035.

Labels remain human under EQUIVARIANT_EDGE_FACE_V1. Not training GT.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
GATE = REPO / "experiments/phase6e/G3_PRESENTATION_EQUIVARIANCE_REPAIR_001.json"
AS2 = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_001.json"
WS_ROOT = REPO / "experiments/phase6e/different-class-discovery-packages"
Q_ROOT = REPO / "experiments/phase6e/different-class-discovery-queues"
BATCH_OUT = REPO / "experiments/phase6e/DIFFERENT_CLASS_DISCOVERY_001.json"
SEALED = REPO / "experiments/phase6e/DIFFERENT_CLASS_DISCOVERY_AXIS_MAP_SEALED_001.json"
SCREEN = REPO / "experiments/phase6e/DIFFERENT_CLASS_DISCOVERY_SCREEN_001.json"

BATCH_ID = "DIFFERENT_CLASS_DISCOVERY_001"
SEED = "DIFFERENT_CLASS_DISCOVERY_001|ABS_FACE_STEP|EQUIVARIANT_EDGE_FACE_V1"
ABS_MIN = 12.0
N_PER_AXIS = 4  # 12 edges total, axis-balanced
SEP = 16
MARGIN = 5
# Prefer C/E-magnitude tissue; 035 already has one low-contrast AS2 location.
PRIMARY = ["MV-DVID-RAW-SURVEY-035"]
SECONDARY = [
    "MV-DVID-RAW-SURVEY-045",
    "MV-DVID-RAW-SURVEY-046",
    "MV-DVID-RAW-SURVEY-047",
    "MV-DVID-RAW-SURVEY-048",
]


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def opaque_id(seq: int) -> str:
    return f"MV-DCD-{_sha(BATCH_ID, f'{seq:04d}')[:12].upper()}"


def face_deltas(a: np.ndarray, axis: int) -> tuple[np.ndarray, tuple]:
    """Return abs |Δ| array and the left-voxel origin offset in the full volume."""
    z, y, x = a.shape
    m = MARGIN
    if axis == 0:
        d = np.abs(a[m + 1 : z - m, m : y - m, m : x - m] - a[m : z - m - 1, m : y - m, m : x - m])
        origin = (m, m, m)
    elif axis == 1:
        d = np.abs(a[m : z - m, m + 1 : y - m, m : x - m] - a[m : z - m, m : y - m - 1, m : x - m])
        origin = (m, m, m)
    else:
        d = np.abs(a[m : z - m, m : y - m, m + 1 : x - m] - a[m : z - m, m : y - m, m : x - m - 1])
        origin = (m, m, m)
    return d, origin


def collect_candidates(sid: str, arr: np.ndarray, exclude_centers: list[tuple[int, int, int]]) -> dict[str, list]:
    out = {"Z": [], "Y": [], "X": []}
    names = ("Z", "Y", "X")
    for ax, name in enumerate(names):
        d, origin = face_deltas(arr, ax)
        ys = np.argwhere(d >= ABS_MIN)
        scored = []
        for p in ys:
            left = (
                int(origin[0] + p[0]),
                int(origin[1] + p[1]),
                int(origin[2] + p[2]),
            )
            if any(
                (left[0] - c[0]) ** 2 + (left[1] - c[1]) ** 2 + (left[2] - c[2]) ** 2 < SEP**2
                for c in exclude_centers
            ):
                continue
            step = float(d[tuple(p)])
            # membrane-likeness proxy: darker endpoint (EM membranes are dark)
            right = list(left)
            right[ax] += 1
            i0 = float(arr[left])
            i1 = float(arr[tuple(right)])
            dark = min(i0, i1)
            scored.append((step, -dark, left, right, i0, i1))
        # rank: higher step, then darker
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        out[name] = scored
    return out


def greedy_pick(cands_by_axis: dict[str, list], n_per: int) -> list[dict]:
    selected: list[dict] = []
    used: list[tuple[int, int, int]] = []
    for name in ("Z", "Y", "X"):
        taken = 0
        for step, neg_dark, left, right, i0, i1 in cands_by_axis[name]:
            if any(
                (left[0] - u[0]) ** 2 + (left[1] - u[1]) ** 2 + (left[2] - u[2]) ** 2 < SEP**2
                for u in used
            ):
                continue
            ax = {"Z": 0, "Y": 1, "X": 2}[name]
            selected.append(
                {
                    "axis_name": name,
                    "channel_zyx": ax,
                    "pair_left_zyx": list(left),
                    "pair_right_zyx": list(right),
                    "abs_face_step": step,
                    "intensity_left": i0,
                    "intensity_right": i1,
                    "darker_endpoint": min(i0, i1),
                }
            )
            used.append(left)
            taken += 1
            if taken >= n_per:
                break
        if taken < n_per:
            raise SystemExit(f"Only {taken} separated abs>={ABS_MIN} edges on axis {name}")
    return selected


def build() -> dict:
    if any(p.exists() for p in (WS_ROOT, Q_ROOT, BATCH_OUT, SEALED, SCREEN)):
        raise FileExistsError("Refusing to overwrite DIFFERENT-class discovery artifacts")
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    if not gate.get("overall_pass"):
        raise SystemExit("Synthetic gate failed")
    as2 = json.loads(AS2.read_text(encoding="utf-8"))
    exclude = []
    for loc in as2["per_location"]:
        if loc["source_id"] == "MV-DVID-RAW-SURVEY-035":
            exclude.append(tuple(loc["center_zyx"]))

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = {r["id"]: r for r in manifest["records"]}

    # Prefer PRIMARY (035) which has balanced high-abs edges on all axes.
    screen_report = {"abs_min": ABS_MIN, "per_source": {}}
    picks = None
    chosen_source = None
    for sid in PRIMARY + SECONDARY:
        arr = np.asarray(np.load(records[sid]["raw_path"], mmap_mode="r", allow_pickle=False), dtype=np.float64)
        excl = exclude if sid == "MV-DVID-RAW-SURVEY-035" else []
        cands = collect_candidates(sid, arr, excl)
        screen_report["per_source"][sid] = {ax: len(v) for ax, v in cands.items()}
        try:
            picks = greedy_pick(cands, N_PER_AXIS)
            chosen_source = sid
            break
        except SystemExit:
            continue
    if picks is None or chosen_source is None:
        raise SystemExit("No source yielded axis-balanced abs>=12 candidates")

    # Attach source metadata
    for p in picks:
        p["source_id"] = chosen_source
        p["raw_sha256"] = records[chosen_source]["raw_sha256"]

    screen_report["chosen_source"] = chosen_source
    screen_report["n_selected"] = len(picks)
    screen_report["as2_excluded_centers_on_035"] = [list(c) for c in exclude]
    screen_report["selection_rule"] = (
        f"abs |ΔI| >= {ABS_MIN} on single face; darker-endpoint tiebreak; "
        f"sep>={SEP}; {N_PER_AXIS} per axis; prefer SURVEY-035 C/E-magnitude tissue"
    )
    screen_report["why_not_as2_relative_q"] = (
        "AS2 relative Q strata selected mean3 face-step in [0.33, 3.0]; "
        "historical C/E abs magnitudes ~8–17. Relative enrichment was insufficient "
        "to place reviewers on membrane-scale transitions."
    )
    SCREEN.write_text(json.dumps(screen_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    n = len(picks)
    order = sorted(range(n), key=lambda i: _sha(SEED, "ORDER", f"{i:02d}"))
    ordered = [picks[i] for i in order]
    for seq, d in enumerate(ordered):
        d["opaque_decision_id"] = opaque_id(seq)
        d["presentation_order_index"] = seq

    order_hash = _sha(SEED, "ORDER_DIGEST", ",".join(d["opaque_decision_id"] for d in ordered))
    mapping_hash = _sha(
        SEED,
        "MAP",
        json.dumps([(d["opaque_decision_id"], d["axis_name"], d["pair_left_zyx"]) for d in ordered], sort_keys=True),
    )

    WS_ROOT.mkdir(parents=True)
    Q_ROOT.mkdir(parents=True)
    r = records[chosen_source]
    crop_id = "MV-DCD-01"
    crop_dir = WS_ROOT / crop_id
    crop_dir.mkdir(parents=True)
    log = crop_dir / "workspace.events.jsonl"
    log.write_text("", encoding="utf-8")
    ws_id = f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id}-DCD"
    workspace = {
        "allowed_decisions": ["BAD_QUESTION", "DIFFERENT_PROCESS", "SAME_PROCESS", "UNCERTAIN"],
        "coordinate_frame": "MV-FRAME-DVID-WASP5-001",
        "created_at": _now(),
        "crop_id": crop_id,
        "event_log": {"append_only": True, "path": str(log.resolve())},
        "id": ws_id,
        "pair_contract": "pair_left_zyx is followed by its positive neighbour along channel_zyx: 0=Z, 1=Y, 2=X.",
        "parent_region_id": chosen_source,
        "presentation_mode": "EQUIVARIANT_EDGE_FACE_V1",
        "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
        "raw": {
            "path": r["raw_path"],
            "sha256": r["raw_sha256"],
            "shape_zyx": r["shape_zyx"],
            "dtype": r["dtype"],
        },
        "provenance": {
            "experiment": BATCH_ID,
            "target_spec": "MV-G3-AFFINITY-TARGET-SPEC-002",
            "not_a_g3_training_cohort": True,
            "not_experiment_009": True,
            "do_not_train": True,
            "discovery_only": True,
            "selection": "ABS_FACE_STEP_GE_12_AXIS_BALANCED",
        },
        "review_state": "UNREVIEWED",
        "schema_version": 1,
        "source_origin_xyz": [r["bounds_xyz"][a][0] for a in ("x", "y", "z")],
        "split": "DIFFERENT_CLASS_DISCOVERY_ONLY",
        "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
    }
    (crop_dir / "workspace.json").write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sealed_entries = []
    master_questions = []
    for d in ordered:
        sealed_entries.append(
            {
                "opaque_decision_id": d["opaque_decision_id"],
                "presentation_order_index": d["presentation_order_index"],
                "crop_id": crop_id,
                "source_id": chosen_source,
                "axis_name": d["axis_name"],
                "channel_zyx": d["channel_zyx"],
                "pair_left_zyx": d["pair_left_zyx"],
                "pair_right_zyx": d["pair_right_zyx"],
                "abs_face_step": d["abs_face_step"],
                "workspace_path": str((crop_dir / "workspace.json").resolve()),
            }
        )
        master_questions.append(
            {
                "opaque_decision_id": d["opaque_decision_id"],
                "presentation_order_index": d["presentation_order_index"],
                "kind": "EQUIVARIANT_EDGE_FACE",
                "pair_left_zyx": d["pair_left_zyx"],
                "pair_right_zyx": d["pair_right_zyx"],
                "crop_id": crop_id,
                "workspace_path": str((crop_dir / "workspace.json").resolve()),
                "model_navigation": False,
            }
        )

    master = {
        "schema_version": 1,
        "id": f"{BATCH_ID}-MASTER-QUEUE",
        "discovery_batch_id": BATCH_ID,
        "created_at": _now(),
        "status": "EXPERT_EQUIVARIANT_DISCOVERY_REVIEW_REQUIRED",
        "axis_blind": True,
        "expected_n_decisions": n,
        "questions": master_questions,
        "scientific_boundary": (
            "DIFFERENT-class discovery under abs face-step enrichment. "
            "Not training. Not G3-009. Not AS2 revalidation."
        ),
    }
    master_path = Q_ROOT / "MASTER_BATCH.json"
    master_path.write_text(json.dumps(master, indent=2) + "\n", encoding="utf-8")

    SEALED.write_text(
        json.dumps(
            {
                "id": "DIFFERENT_CLASS_DISCOVERY_AXIS_MAP_SEALED_001",
                "status": f"SEALED_UNTIL_ALL_{n}_DECISIONS_COMPLETE",
                "batch_id": BATCH_ID,
                "seed_material": SEED,
                "order_hash": order_hash,
                "mapping_hash": mapping_hash,
                "entries": sealed_entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    batch = {
        "id": BATCH_ID,
        "schema_version": 1,
        "status": "PROVISIONED_AWAITING_HUMAN_DECISIONS",
        "created_at": _now(),
        "target_spec": "MV-G3-AFFINITY-TARGET-SPEC-002",
        "presentation_mode": "EQUIVARIANT_EDGE_FACE_V1",
        "not_a_g3_training_cohort": True,
        "not_experiment_009": True,
        "do_not_train": True,
        "not_as2_revalidation": True,
        "synthetic_gate": {"overall_pass": True, "path": str(GATE.relative_to(REPO)).replace("\\", "/")},
        "design": {
            "n_decisions": n,
            "n_per_axis": N_PER_AXIS,
            "abs_face_step_min": ABS_MIN,
            "source": chosen_source,
            "selection": screen_report["selection_rule"],
            "screen_path": str(SCREEN.relative_to(REPO)).replace("\\", "/"),
            "note_on_035": (
                "SURVEY-035 was used in AS2 at one low-contrast center (mean3≈2). "
                "This discovery uses other centers with abs face-step ≥12, spatially "
                f"separated by ≥{SEP} voxels from that AS2 center."
            ),
        },
        "blinding": {
            "axis_blind_during_judgment": True,
            "order_hash": order_hash,
            "mapping_hash": mapping_hash,
            "sealed_axis_map_path": str(SEALED.relative_to(REPO)).replace("\\", "/"),
        },
        "predeclared_verdict_rules": {
            "DIFFERENT_CLASS_DEMONSTRATED": "≥1 DIFFERENT_PROCESS among class-countable edges.",
            "DIFFERENT_CLASS_STILL_NOT_FOUND": "Zero DIFFERENT_PROCESS among class-countable edges.",
            "exclusions_from_class_counts": ["UNCERTAIN", "BAD_QUESTION"],
            "priority_order": ["DIFFERENT_CLASS_STILL_NOT_FOUND", "DIFFERENT_CLASS_DEMONSTRATED"],
            "do_not_train": True,
            "discovery_labels_are_not_automatic_production_gt": True,
        },
        "paths": {
            "packages": str(WS_ROOT.relative_to(REPO)).replace("\\", "/"),
            "queues": str(Q_ROOT.relative_to(REPO)).replace("\\", "/"),
            "master_queue": str(master_path.relative_to(REPO)).replace("\\", "/"),
        },
        "launch_example": (
            r'.\.venv-reviewer\Scripts\python.exe tools\review_equivariant_edge_batch.py '
            r'experiments\phase6e\different-class-discovery-queues\MASTER_BATCH.json '
            r'--reviewer matth'
        ),
        "results": None,
        "verdict": None,
    }
    BATCH_OUT.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return batch


if __name__ == "__main__":
    b = build()
    print(
        json.dumps(
            {
                "id": b["id"],
                "source": b["design"]["source"],
                "n": b["design"]["n_decisions"],
                "abs_min": b["design"]["abs_face_step_min"],
                "order_hash": b["blinding"]["order_hash"],
            },
            indent=2,
        )
    )
