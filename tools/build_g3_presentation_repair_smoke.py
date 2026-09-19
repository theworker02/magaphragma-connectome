"""Provision the presentation-repair human smoke experiment (NOT a G3 cohort / NOT -009).

Requires G3_PRESENTATION_EQUIVARIANCE_REPAIR_001.json overall_pass=true.
Selects 4 fresh physical locations on unused survey crops 030/032 (2 each),
axis-blinds all 12 decisions with deterministic pseudorandom order, and freezes
predeclared interpretation rules BEFORE any human labels.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import build_g3_008_axis_neutral as g8  # reuse center geometry helpers only

MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
GATE = REPO / "experiments/phase6e/G3_PRESENTATION_EQUIVARIANCE_REPAIR_001.json"
WS_ROOT = REPO / "experiments/phase6e/g3-presentation-repair-smoke-packages"
Q_ROOT = REPO / "experiments/phase6e/g3-presentation-repair-smoke-queues"
SMOKE_OUT = REPO / "experiments/phase6e/G3_PRESENTATION_REPAIR_SMOKE_001.json"
SEALED_MAP = REPO / "experiments/phase6e/G3_PRESENTATION_REPAIR_SMOKE_AXIS_MAP_SEALED_001.json"

# Fresh crops never used by -005/-006/-008; C/E excluded from survey eligible already.
SMOKE_SOURCES = ["MV-DVID-RAW-SURVEY-030", "MV-DVID-RAW-SURVEY-032"]
LOCATIONS_PER_SOURCE = 2
SEED_MATERIAL = "MV-G3-PRESENTATION-REPAIR-SMOKE-001|SPEC_002|EQUIVARIANT_EDGE_FACE_V1"
BATCH_ID = "MV-G3-PRESENTATION-REPAIR-SMOKE-001"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def require_gate() -> dict:
    if not GATE.exists():
        raise SystemExit("Missing equivariance repair gate artifact; run g3_presentation_equivariance_repair_gate.py")
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    if not gate.get("overall_pass"):
        raise SystemExit("SYNTHETIC GATE FAILED — refuse to provision human smoke")
    return gate


def pick_centers(arr: np.ndarray, n: int, source_id: str) -> list[tuple[int, int, int]]:
    coords, cvals = g8.eligible_centers(arr)
    kept = g8.greedy_separated_centers(coords, cvals)
    # Deterministic rank by hash, not by contrast (avoid label-like targeting)
    ranked = sorted(kept, key=lambda item: _sha(SEED_MATERIAL, source_id, str(item[0])))
    if len(ranked) < n:
        raise SystemExit(f"{source_id}: only {len(ranked)} separated centers, need {n}")
    return [ranked[i][0] for i in range(n)]


def opaque_id(batch: str, seq: int) -> str:
    return f"MV-PRS-{_sha(batch, f'{seq:04d}')[:12].upper()}"


def build() -> dict:
    gate = require_gate()
    if WS_ROOT.exists() or Q_ROOT.exists() or SMOKE_OUT.exists():
        raise FileExistsError("Refusing to overwrite existing presentation-repair smoke artifacts")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rec = {r["id"]: r for r in manifest["records"]}
    for sid in SMOKE_SOURCES:
        if sid not in rec:
            raise SystemExit(f"missing survey record {sid}")

    # Exclude exact historical edges from pinned snapshot (geometry safety)
    hist_list, hist_hash = g8.pinned_historical_edges()
    hist_set = {(tuple(l), tuple(r), c) for (l, r, c) in hist_list}

    locations = []
    for sid in SMOKE_SOURCES:
        arr = np.load(rec[sid]["raw_path"], mmap_mode="r", allow_pickle=False)
        for center in pick_centers(arr, LOCATIONS_PER_SOURCE, sid):
            edges = g8._edges(center)
            conflict = False
            for ax, (left, right) in edges.items():
                if (tuple(left), tuple(right), g8._axis_index(ax)) in hist_set:
                    conflict = True
            if conflict:
                raise SystemExit(f"{sid} center {center} collides with pinned historical edge")
            locations.append({"source_id": sid, "center_zyx": list(center), "raw_sha256": rec[sid]["raw_sha256"]})

    assert len(locations) == 4

    # 12 decisions: expand then blind-shuffle with predeclared seed
    decisions = []
    for loc_i, loc in enumerate(locations):
        for ax in ("Z", "Y", "X"):
            left, right = g8._edges(tuple(loc["center_zyx"]))[ax]
            decisions.append({
                "source_id": loc["source_id"],
                "center_zyx": loc["center_zyx"],
                "raw_sha256": loc["raw_sha256"],
                "axis_name": ax,
                "channel_zyx": g8._axis_index(ax),
                "pair_left_zyx": list(left),
                "pair_right_zyx": list(right),
                "location_index": loc_i,
            })

    order = sorted(range(12), key=lambda i: _sha(SEED_MATERIAL, "ORDER", f"{i:02d}"))
    ordered = [decisions[i] for i in order]
    for seq, d in enumerate(ordered):
        d["opaque_decision_id"] = opaque_id(BATCH_ID, seq)
        d["presentation_order_index"] = seq

    order_hash = _sha(SEED_MATERIAL, "ORDER_DIGEST", ",".join(d["opaque_decision_id"] for d in ordered))
    mapping_hash = _sha(
        SEED_MATERIAL,
        "MAP",
        json.dumps([(d["opaque_decision_id"], d["axis_name"], d["center_zyx"]) for d in ordered], sort_keys=True),
    )

    WS_ROOT.mkdir(parents=True)
    Q_ROOT.mkdir(parents=True)

    # One workspace+queue per location (3 edges each); questions inside queue are
    # globally ordered by presentation_order_index across the batch file list.
    sealed_entries = []
    per_location = []
    for loc_i, loc in enumerate(locations):
        crop_id = f"MV-G3-PRSMOKE-{loc_i + 1:02d}"
        sid = loc["source_id"]
        r = rec[sid]
        crop_dir = WS_ROOT / crop_id
        crop_dir.mkdir(parents=True)
        log = crop_dir / "workspace.events.jsonl"
        log.write_text("", encoding="utf-8")
        ws_id = f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id}-PRSMOKE"
        workspace = {
            "allowed_decisions": ["BAD_QUESTION", "DIFFERENT_PROCESS", "SAME_PROCESS", "UNCERTAIN"],
            "coordinate_frame": "MV-FRAME-DVID-WASP5-001",
            "created_at": _now(),
            "crop_id": crop_id,
            "event_log": {"append_only": True, "path": str(log.resolve())},
            "id": ws_id,
            "pair_contract": "pair_left_zyx is followed by its positive neighbour along channel_zyx: 0=Z, 1=Y, 2=X.",
            "parent_region_id": sid,
            "presentation_mode": "EQUIVARIANT_EDGE_FACE_V1",
            "prohibited_promotions": ["REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
            "raw": {"path": r["raw_path"], "sha256": r["raw_sha256"], "shape_zyx": r["shape_zyx"], "dtype": r["dtype"]},
            "provenance": {
                "experiment": BATCH_ID,
                "target_spec": "MV-G3-AFFINITY-TARGET-SPEC-002",
                "not_a_g3_training_cohort": True,
                "not_experiment_009": True,
                "center_zyx": loc["center_zyx"],
            },
            "review_state": "UNREVIEWED",
            "schema_version": 1,
            "source_origin_xyz": [r["bounds_xyz"][a][0] for a in ("x", "y", "z")],
            "split": "PRESENTATION_REPAIR_SMOKE_ONLY",
            "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
        }
        (crop_dir / "workspace.json").write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        loc_questions = [d for d in ordered if d["location_index"] == loc_i]
        loc_questions.sort(key=lambda d: d["presentation_order_index"])
        questions = []
        for d in loc_questions:
            questions.append({
                "opaque_decision_id": d["opaque_decision_id"],
                "presentation_order_index": d["presentation_order_index"],
                "kind": "EQUIVARIANT_EDGE_FACE",
                "pair_left_zyx": d["pair_left_zyx"],
                "pair_right_zyx": d["pair_right_zyx"],
                # channel/axis omitted from reviewer-facing queue; derived from pair geometry.
                "model_navigation": False,
            })
            sealed_entries.append({
                "opaque_decision_id": d["opaque_decision_id"],
                "presentation_order_index": d["presentation_order_index"],
                "crop_id": crop_id,
                "source_id": sid,
                "center_zyx": d["center_zyx"],
                "axis_name": d["axis_name"],
                "channel_zyx": d["channel_zyx"],
                "pair_left_zyx": d["pair_left_zyx"],
                "pair_right_zyx": d["pair_right_zyx"],
                "workspace_path": str((crop_dir / "workspace.json").resolve()),
            })

        queue = {
            "schema_version": 1,
            "id": f"MV-G3-PRSMOKE-QUEUE-{crop_id}",
            "smoke_batch_id": BATCH_ID,
            "created_at": _now(),
            "status": "EXPERT_EQUIVARIANT_SMOKE_REVIEW_REQUIRED",
            "axis_blind": True,
            "workspace_id": ws_id,
            "crop_id": crop_id,
            "raw_sha256": r["raw_sha256"],
            "questions": questions,
            "scientific_boundary": (
                "Presentation-repair smoke only under SPEC_002. Not training supervision. "
                "Not G3-009. Axis identity hidden until batch freeze."
            ),
        }
        (Q_ROOT / f"{crop_id}.json").write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
        per_location.append({
            "crop_id": crop_id,
            "source_id": sid,
            "center_zyx": loc["center_zyx"],
            "workspace_id": ws_id,
            "n_questions": 3,
        })

    # Master batch queue: all 12 decisions in blinded global order (cross-location).
    master_questions = []
    for d in ordered:
        crop_id = f"MV-G3-PRSMOKE-{d['location_index'] + 1:02d}"
        master_questions.append({
            "opaque_decision_id": d["opaque_decision_id"],
            "presentation_order_index": d["presentation_order_index"],
            "kind": "EQUIVARIANT_EDGE_FACE",
            "pair_left_zyx": d["pair_left_zyx"],
            "pair_right_zyx": d["pair_right_zyx"],
            "crop_id": crop_id,
            "workspace_path": str((WS_ROOT / crop_id / "workspace.json").resolve()),
            "model_navigation": False,
        })
    master = {
        "schema_version": 1,
        "id": f"{BATCH_ID}-MASTER-QUEUE",
        "smoke_batch_id": BATCH_ID,
        "created_at": _now(),
        "status": "EXPERT_EQUIVARIANT_SMOKE_REVIEW_REQUIRED",
        "axis_blind": True,
        "questions": master_questions,
        "scientific_boundary": (
            "All 12 smoke decisions in deterministic blinded order across 4 locations. "
            "Not training supervision. Not G3-009."
        ),
    }
    master_path = Q_ROOT / "MASTER_BATCH.json"
    master_path.write_text(json.dumps(master, indent=2) + "\n", encoding="utf-8")


    sealed = {
        "id": "MV-G3-PRESENTATION-REPAIR-SMOKE-AXIS-MAP-SEALED-001",
        "status": "SEALED_UNTIL_ALL_12_DECISIONS_COMPLETE",
        "batch_id": BATCH_ID,
        "seed_material": SEED_MATERIAL,
        "order_hash": order_hash,
        "mapping_hash": mapping_hash,
        "entries": sealed_entries,
    }
    SEALED_MAP.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    smoke = {
        "id": BATCH_ID,
        "schema_version": 1,
        "status": "PROVISIONED_AWAITING_HUMAN_DECISIONS",
        "created_at": _now(),
        "target_spec": "MV-G3-AFFINITY-TARGET-SPEC-002",
        "presentation_mode": "EQUIVARIANT_EDGE_FACE_V1",
        "not_a_g3_training_cohort": True,
        "not_experiment_009": True,
        "do_not_train": True,
        "synthetic_gate": {
            "path": str(GATE.relative_to(REPO)).replace("\\", "/"),
            "overall_pass": True,
            "fixtures_sha256": gate.get("fixtures_sha256"),
        },
        "design": {
            "n_locations": 4,
            "edges_per_location": 3,
            "total_decisions": 12,
            "sources": SMOKE_SOURCES,
            "selection_rule": "hash-ranked greedy-separated centers; no label/model inputs",
            "excluded": ["C/E SURVEY-017/027", "all -008 locations/sources", "prior -005/-006 sources"],
        },
        "blinding": {
            "axis_blind_during_judgment": True,
            "fixed_ZYX_order": False,
            "deterministic_pseudorandom_order_seed_material": SEED_MATERIAL,
            "order_hash": order_hash,
            "mapping_hash": mapping_hash,
            "sealed_axis_map_path": str(SEALED_MAP.relative_to(REPO)).replace("\\", "/"),
            "unblind_only_after_all_12_complete": True,
        },
        "predeclared_interpretation": {
            "frozen_before_labels": True,
            "question": (
                "Does presentation repair eliminate or materially disrupt the deterministic "
                "orientation-collinear response (Z=DIFFERENT, Y=SAME, X=SAME) observed under the invalid interface?"
            ),
            "pathological_pattern": {
                "Z": "DIFFERENT_PROCESS",
                "Y": "SAME_PROCESS",
                "X": "SAME_PROCESS",
            },
            "verdict_rules": {
                "PRESENTATION_FIX_INSUFFICIENT": (
                    "After unblinding, all 4 location triplets exactly match the pathological pattern "
                    "(or all completed triplets do, if any UNCERTAIN/BAD excluded from pattern test and "
                    "remaining still all match)."
                ),
                "PRESENTATION_DEFECT_CAUSALLY_SUPPORTED": (
                    "After unblinding, the set of location triplets is not uniformly the pathological "
                    "pattern (at least one location breaks Z=DIFF & Y=SAME & X=SAME)."
                ),
            },
            "exclusions_from_pattern_test": ["UNCERTAIN", "BAD_QUESTION"],
            "smoke_labels_are_not_production_gt": True,
            "no_interface_tuning_after_seeing_labels": True,
        },
        "paths": {
            "packages": str(WS_ROOT.relative_to(REPO)).replace("\\", "/"),
            "queues": str(Q_ROOT.relative_to(REPO)).replace("\\", "/"),
            "master_queue": "experiments/phase6e/g3-presentation-repair-smoke-queues/MASTER_BATCH.json",
        },
        "per_location": per_location,
        "launch_example": (
            r'.\.venv-reviewer\Scripts\python.exe tools\review_equivariant_edge_batch.py '
            r'experiments\phase6e\g3-presentation-repair-smoke-queues\MASTER_BATCH.json '
            r'--reviewer matth'
        ),
        "results": None,
        "verdict": None,
    }
    SMOKE_OUT.write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return smoke


if __name__ == "__main__":
    s = build()
    print(json.dumps({
        "id": s["id"],
        "status": s["status"],
        "locations": len(s["per_location"]),
        "order_hash": s["blinding"]["order_hash"],
        "mapping_hash": s["blinding"]["mapping_hash"],
    }, indent=2))
