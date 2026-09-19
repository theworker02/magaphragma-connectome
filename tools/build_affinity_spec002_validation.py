"""Provision AFFINITY_SPEC002_VALIDATION_001 (NOT -009, NOT training).

Requires:
  - synthetic equivariance gate overall_pass
  - protocol status FROZEN_READY_TO_PROVISION
  - ≥12 never-reviewed independent sources in inventory

Selects 12 sources (prefer newly acquired SURVEY-033+), one location each,
relative Q1–Q4 strata, axis-blind 36 decisions, sealed axis map.
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
import build_g3_008_axis_neutral as g8

PROTOCOL = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_PROTOCOL_001.json"
INVENTORY = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_SOURCE_INVENTORY_001.json"
MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
GATE = REPO / "experiments/phase6e/G3_PRESENTATION_EQUIVARIANCE_REPAIR_001.json"
ACQUISITION = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_SOURCE_ACQUISITION_001.json"

WS_ROOT = REPO / "experiments/phase6e/affinity-spec002-validation-packages"
Q_ROOT = REPO / "experiments/phase6e/affinity-spec002-validation-queues"
BATCH_OUT = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_001.json"
SEALED_MAP = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_AXIS_MAP_SEALED_001.json"

N_LOCATIONS = 12
STRATA = ["Q1", "Q2", "Q3", "Q4"]
PERCENTILE_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]
SEED_MATERIAL = "AFFINITY_SPEC002_VALIDATION_PROTOCOL_001|SPEC_002|EQUIVARIANT_EDGE_FACE_V1"
BATCH_ID = "AFFINITY_SPEC002_VALIDATION_001"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _hash_rank(*parts: str) -> int:
    return int(_sha(*parts), 16)


def opaque_id(seq: int) -> str:
    return f"MV-AS2-{_sha(BATCH_ID, f'{seq:04d}')[:12].upper()}"


def choose_sources(eligible: list[str], newly_acquired: set[str]) -> list[str]:
    preferred = sorted(s for s in eligible if s in newly_acquired)
    remainder = sorted(s for s in eligible if s not in newly_acquired)
    ordered = preferred + remainder
    if len(ordered) < N_LOCATIONS:
        raise SystemExit(f"Only {len(ordered)} eligible sources; need {N_LOCATIONS}")
    return ordered[:N_LOCATIONS]


def select_locations(sources: list[str], records: dict) -> tuple[list[dict], list[float]]:
    hist_list, _ = g8.pinned_historical_edges()
    hist_set = {(tuple(l), tuple(r), c) for (l, r, c) in hist_list}

    crop_centers: dict[str, list] = {}
    all_contrasts: list[float] = []
    for sid in sources:
        arr = np.load(records[sid]["raw_path"], mmap_mode="r", allow_pickle=False)
        centers = g8.greedy_separated_centers(*g8.eligible_centers(arr))
        crop_centers[sid] = centers
        all_contrasts.extend(ct for _, ct in centers)
    if not all_contrasts:
        raise SystemExit("No eligible centers in selected sources")
    bounds = [float(np.quantile(np.array(all_contrasts, dtype=np.float64), p)) for p in PERCENTILE_EDGES]

    def stratum_of(ct: float) -> str:
        for i in range(4):
            lo, hi = bounds[i], bounds[i + 1]
            if (lo <= ct < hi) or (i == 3 and ct <= hi):
                return STRATA[i]
        return STRATA[-1]

    assignments = [STRATA[i % 4] for i in range(N_LOCATIONS)]
    selection = []
    for i, sid in enumerate(sources):
        ts = assignments[i]
        raw_sha = records[sid]["raw_sha256"]
        cands = []
        for (c, ct) in crop_centers[sid]:
            if stratum_of(ct) != ts:
                continue
            edges = g8._edges(c)
            if any(
                (tuple(edges[ax][0]), tuple(edges[ax][1]), g8._axis_index(ax)) in hist_set
                for ax in ("Z", "Y", "X")
            ):
                continue
            cands.append((c, ct))
        if not cands:
            raise SystemExit(f"{sid}: no eligible center in stratum {ts} after exclusion")
        pick = min(cands, key=lambda cc: _hash_rank(SEED_MATERIAL, raw_sha, sid, ts, str(cc[0])))
        selection.append(
            {
                "source_id": sid,
                "stratum": ts,
                "center_zyx": list(pick[0]),
                "raw_contrast": pick[1],
                "raw_sha256": raw_sha,
            }
        )
    return selection, bounds


def build() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_READY_TO_PROVISION":
        raise SystemExit(f"Protocol not ready: {protocol.get('status')}")
    if protocol.get("do_not_train") is not True or protocol.get("not_g3_009") is not True:
        raise SystemExit("Protocol missing do_not_train / not_g3_009 guards")

    gate = json.loads(GATE.read_text(encoding="utf-8"))
    if not gate.get("overall_pass"):
        raise SystemExit("SYNTHETIC GATE FAILED — refuse to provision validation")

    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    if not inventory["independence_requirement"]["satisfied"]:
        raise SystemExit("Independence unsatisfied; refuse provision")

    if WS_ROOT.exists() or Q_ROOT.exists() or BATCH_OUT.exists() or SEALED_MAP.exists():
        raise FileExistsError("Refusing to overwrite existing SPEC002 validation artifacts")

    acquisition = json.loads(ACQUISITION.read_text(encoding="utf-8"))
    newly = {r["id"] for r in acquisition["records"]}
    eligible = inventory["eligible_independent_never_reviewed"]
    sources = choose_sources(eligible, newly)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = {r["id"]: r for r in manifest["records"]}
    for sid in sources:
        if sid not in records:
            raise SystemExit(f"missing survey record {sid}")

    selection, bounds = select_locations(sources, records)
    assert len(selection) == N_LOCATIONS

    decisions = []
    for loc_i, loc in enumerate(selection):
        for ax in ("Z", "Y", "X"):
            left, right = g8._edges(tuple(loc["center_zyx"]))[ax]
            decisions.append(
                {
                    "source_id": loc["source_id"],
                    "center_zyx": loc["center_zyx"],
                    "raw_sha256": loc["raw_sha256"],
                    "stratum": loc["stratum"],
                    "axis_name": ax,
                    "channel_zyx": g8._axis_index(ax),
                    "pair_left_zyx": list(left),
                    "pair_right_zyx": list(right),
                    "location_index": loc_i,
                }
            )

    n_dec = N_LOCATIONS * 3
    order = sorted(range(n_dec), key=lambda i: _sha(SEED_MATERIAL, "ORDER", f"{i:02d}"))
    ordered = [decisions[i] for i in order]
    for seq, d in enumerate(ordered):
        d["opaque_decision_id"] = opaque_id(seq)
        d["presentation_order_index"] = seq

    order_hash = _sha(SEED_MATERIAL, "ORDER_DIGEST", ",".join(d["opaque_decision_id"] for d in ordered))
    mapping_hash = _sha(
        SEED_MATERIAL,
        "MAP",
        json.dumps(
            [(d["opaque_decision_id"], d["axis_name"], d["center_zyx"]) for d in ordered],
            sort_keys=True,
        ),
    )

    WS_ROOT.mkdir(parents=True)
    Q_ROOT.mkdir(parents=True)

    sealed_entries = []
    per_location = []
    for loc_i, loc in enumerate(selection):
        crop_id = f"MV-AS2-VAL-{loc_i + 1:02d}"
        sid = loc["source_id"]
        r = records[sid]
        crop_dir = WS_ROOT / crop_id
        crop_dir.mkdir(parents=True)
        log = crop_dir / "workspace.events.jsonl"
        log.write_text("", encoding="utf-8")
        ws_id = f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id}-AS2VAL"
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
                "center_zyx": loc["center_zyx"],
                "stratum": loc["stratum"],
            },
            "review_state": "UNREVIEWED",
            "schema_version": 1,
            "source_origin_xyz": [r["bounds_xyz"][a][0] for a in ("x", "y", "z")],
            "split": "AFFINITY_SPEC002_VALIDATION_ONLY",
            "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
        }
        (crop_dir / "workspace.json").write_text(
            json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        loc_questions = [d for d in ordered if d["location_index"] == loc_i]
        loc_questions.sort(key=lambda d: d["presentation_order_index"])
        questions = []
        for d in loc_questions:
            questions.append(
                {
                    "opaque_decision_id": d["opaque_decision_id"],
                    "presentation_order_index": d["presentation_order_index"],
                    "kind": "EQUIVARIANT_EDGE_FACE",
                    "pair_left_zyx": d["pair_left_zyx"],
                    "pair_right_zyx": d["pair_right_zyx"],
                    "model_navigation": False,
                }
            )
            sealed_entries.append(
                {
                    "opaque_decision_id": d["opaque_decision_id"],
                    "presentation_order_index": d["presentation_order_index"],
                    "crop_id": crop_id,
                    "source_id": sid,
                    "center_zyx": d["center_zyx"],
                    "stratum": d["stratum"],
                    "axis_name": d["axis_name"],
                    "channel_zyx": d["channel_zyx"],
                    "pair_left_zyx": d["pair_left_zyx"],
                    "pair_right_zyx": d["pair_right_zyx"],
                    "workspace_path": str((crop_dir / "workspace.json").resolve()),
                }
            )

        queue = {
            "schema_version": 1,
            "id": f"MV-AS2-VAL-QUEUE-{crop_id}",
            "validation_batch_id": BATCH_ID,
            "created_at": _now(),
            "status": "EXPERT_EQUIVARIANT_VALIDATION_REVIEW_REQUIRED",
            "axis_blind": True,
            "workspace_id": ws_id,
            "crop_id": crop_id,
            "raw_sha256": r["raw_sha256"],
            "questions": questions,
            "scientific_boundary": (
                "SPEC002 annotation-procedure validation only. Not training supervision. "
                "Not G3-009. Axis identity hidden until batch freeze."
            ),
        }
        (Q_ROOT / f"{crop_id}.json").write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
        per_location.append(
            {
                "crop_id": crop_id,
                "source_id": sid,
                "center_zyx": loc["center_zyx"],
                "stratum": loc["stratum"],
                "raw_contrast": loc["raw_contrast"],
                "workspace_id": ws_id,
                "n_questions": 3,
            }
        )

    master_questions = []
    for d in ordered:
        crop_id = f"MV-AS2-VAL-{d['location_index'] + 1:02d}"
        master_questions.append(
            {
                "opaque_decision_id": d["opaque_decision_id"],
                "presentation_order_index": d["presentation_order_index"],
                "kind": "EQUIVARIANT_EDGE_FACE",
                "pair_left_zyx": d["pair_left_zyx"],
                "pair_right_zyx": d["pair_right_zyx"],
                "crop_id": crop_id,
                "workspace_path": str((WS_ROOT / crop_id / "workspace.json").resolve()),
                "model_navigation": False,
            }
        )
    master = {
        "schema_version": 1,
        "id": f"{BATCH_ID}-MASTER-QUEUE",
        "validation_batch_id": BATCH_ID,
        "created_at": _now(),
        "status": "EXPERT_EQUIVARIANT_VALIDATION_REVIEW_REQUIRED",
        "axis_blind": True,
        "expected_n_decisions": n_dec,
        "questions": master_questions,
        "scientific_boundary": (
            f"All {n_dec} SPEC002 validation decisions in deterministic blinded order "
            f"across {N_LOCATIONS} locations. Not training. Not G3-009."
        ),
    }
    master_path = Q_ROOT / "MASTER_BATCH.json"
    master_path.write_text(json.dumps(master, indent=2) + "\n", encoding="utf-8")

    sealed = {
        "id": "AFFINITY_SPEC002_VALIDATION_AXIS_MAP_SEALED_001",
        "status": "SEALED_UNTIL_ALL_36_DECISIONS_COMPLETE",
        "batch_id": BATCH_ID,
        "seed_material": SEED_MATERIAL,
        "order_hash": order_hash,
        "mapping_hash": mapping_hash,
        "entries": sealed_entries,
    }
    SEALED_MAP.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    batch = {
        "id": BATCH_ID,
        "schema_version": 1,
        "status": "PROVISIONED_AWAITING_HUMAN_DECISIONS",
        "created_at": _now(),
        "target_spec": "MV-G3-AFFINITY-TARGET-SPEC-002",
        "presentation_mode": "EQUIVARIANT_EDGE_FACE_V1",
        "protocol_id": "AFFINITY_SPEC002_VALIDATION_PROTOCOL_001",
        "not_a_g3_training_cohort": True,
        "not_experiment_009": True,
        "do_not_train": True,
        "smoke_labels_are_not_production_gt": True,
        "synthetic_gate": {
            "path": str(GATE.relative_to(REPO)).replace("\\", "/"),
            "overall_pass": True,
            "fixtures_sha256": gate.get("fixtures_sha256"),
        },
        "design": {
            "n_locations": N_LOCATIONS,
            "edges_per_location": 3,
            "total_decisions": n_dec,
            "sources": sources,
            "one_location_per_source": True,
            "selection_rule": (
                "Prefer newly acquired never-reviewed crops; relative Q1–Q4 strata over pooled "
                "separated-center contrast; hash-min center within stratum; pinned historical edge exclusion."
            ),
            "population_contrast_boundaries": bounds,
            "contrast_strata_semantics": (
                "RELATIVE percentiles Q1–Q4 over pooled separated-center contrast of the 12 "
                "selected fresh sources; NOT absolute and NOT SAME/DIFFERENT-like."
            ),
            "excluded": [
                "008 sources",
                "presentation-repair smoke sources",
                "SURVEY-017/027",
                "any source with nonempty human event log",
            ],
        },
        "blinding": {
            "axis_blind_during_judgment": True,
            "fixed_ZYX_order": False,
            "deterministic_pseudorandom_order_seed_material": SEED_MATERIAL,
            "order_hash": order_hash,
            "mapping_hash": mapping_hash,
            "sealed_axis_map_path": str(SEALED_MAP.relative_to(REPO)).replace("\\", "/"),
            "unblind_only_after_all_36_complete": True,
        },
        "predeclared_verdict_rules": protocol["predeclared_verdict_rules"],
        "paths": {
            "packages": str(WS_ROOT.relative_to(REPO)).replace("\\", "/"),
            "queues": str(Q_ROOT.relative_to(REPO)).replace("\\", "/"),
            "master_queue": str(master_path.relative_to(REPO)).replace("\\", "/"),
        },
        "per_location": per_location,
        "launch_example": (
            r'.\.venv-reviewer\Scripts\python.exe tools\review_equivariant_edge_batch.py '
            r'experiments\phase6e\affinity-spec002-validation-queues\MASTER_BATCH.json '
            r'--reviewer matth'
        ),
        "results": None,
        "verdict": None,
    }
    BATCH_OUT.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    protocol["status"] = "PROVISIONED_AWAITING_HUMAN_DECISIONS"
    protocol["provisioned_at"] = _now()
    protocol["batch_artifact"] = str(BATCH_OUT.relative_to(REPO)).replace("\\", "/")
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return batch


if __name__ == "__main__":
    b = build()
    print(
        json.dumps(
            {
                "id": b["id"],
                "status": b["status"],
                "locations": len(b["per_location"]),
                "sources": b["design"]["sources"],
                "order_hash": b["blinding"]["order_hash"],
                "mapping_hash": b["blinding"]["mapping_hash"],
            },
            indent=2,
        )
    )
