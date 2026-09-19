"""Freeze AFFINITY_SPEC002_VALIDATION_PROTOCOL_001 and assess fresh-source independence.

Fail-closed if synthetic equivariance fails or fewer than N_LOCATIONS never-reviewed
independent sources exist. Does not provision human packages when independence fails.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_PROTOCOL_001.json"
INVENTORY = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_SOURCE_INVENTORY_001.json"
GATE_TOOL = REPO / "tools/g3_presentation_equivariance_repair_gate.py"
GATE_OUT = REPO / "experiments/phase6e/G3_PRESENTATION_EQUIVARIANCE_REPAIR_001.json"
MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
SURVEY = REPO / "experiments/phase6e/MV-G3-NEW-REGION-SURVEY-001.json"
SMOKE = REPO / "experiments/phase6e/G3_PRESENTATION_REPAIR_SMOKE_001.json"
RUN1 = REPO / "bin/run1.json"
PRES_MOD = REPO / "src/mvconnectome/equivariant_edge_presentation.py"
REVIEW_BATCH = REPO / "tools/review_equivariant_edge_batch.py"

N_LOCATIONS = 12  # one per independent never-reviewed source crop
BATCH_ID = "AFFINITY_SPEC002_VALIDATION_001"
SEED = "AFFINITY_SPEC002_VALIDATION_PROTOCOL_001|SPEC_002|EQUIVARIANT_EDGE_FACE_V1"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sources_with_human_decisions() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    roots = [
        REPO / "experiments/phase6e",
        REPO / "local_research_build/phase6e",
    ]
    for root in roots:
        if not root.exists():
            continue
        for log in root.rglob("workspace.events.jsonl"):
            if not log.read_text(encoding="utf-8").strip():
                continue
            ws_path = log.parent / "workspace.json"
            if not ws_path.exists():
                continue
            ws = json.loads(ws_path.read_text(encoding="utf-8"))
            sid = ws.get("parent_region_id")
            if not sid:
                continue
            found.setdefault(sid, []).append(str(log.relative_to(REPO)).replace("\\", "/"))
    return found


def main() -> int:
    if PROTOCOL.exists():
        raise SystemExit(f"Protocol already exists: {PROTOCOL} (refuse overwrite)")

    # --- synthetic gate (absolute) ---
    rc = subprocess.call([sys.executable, str(GATE_TOOL)], cwd=str(REPO))
    gate = json.loads(GATE_OUT.read_text(encoding="utf-8"))
    if rc != 0 or not gate.get("overall_pass"):
        protocol = {
            "id": "AFFINITY_SPEC002_VALIDATION_PROTOCOL_001",
            "status": "STOPPED_SYNTHETIC_EQUIVARIANCE_FAILED",
            "created_at": _now(),
            "synthetic_gate": {"overall_pass": False, "path": str(GATE_OUT.relative_to(REPO)).replace("\\", "/")},
        }
        PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"stop": "synthetic_gate_failed"}, indent=2))
        return 1

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    survey = json.loads(SURVEY.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    run1 = json.loads(RUN1.read_text(encoding="utf-8"))
    reviewed = sources_with_human_decisions()

    smoke_sources = set(smoke["design"]["sources"])
    smoke_centers = {
        (loc["source_id"], tuple(loc["center_zyx"])) for loc in smoke["per_location"]
    }
    run1_sources = set(run1["fresh_sources"])
    run1_centers = {
        (it["source_id"], tuple(it["center_zyx"])) for it in run1["selection"]
    }

    records = {r["id"]: r for r in manifest["records"]}
    never_reviewed = sorted(sid for sid in records if sid not in reviewed)

    # Explicit exclusions (even if somehow never-reviewed)
    hard_exclude = set(smoke_sources) | set(run1_sources)
    hard_exclude |= {
        "MV-DVID-RAW-SURVEY-017",
        "MV-DVID-RAW-SURVEY-027",
    }  # historical C/E
    # Prior human-reviewed A–H / validation crops already in `reviewed`

    eligible_independent = [
        sid
        for sid in never_reviewed
        if sid not in hard_exclude and sid in survey.get("eligible_candidates", [])
    ]
    # never-reviewed but not in eligible list (still usable if not hard-excluded?)
    # Prefer survey-eligible; also allow never-reviewed eligible-or-not if not hard excluded
    # and not overlapping existing G3 regions — for independence we require never_reviewed ∩ not hard_exclude
    eligible_independent = [
        sid for sid in never_reviewed if sid not in hard_exclude
    ]

    inventory = {
        "id": "AFFINITY_SPEC002_VALIDATION_SOURCE_INVENTORY_001",
        "created_at": _now(),
        "survey_manifest_sha256": _sha_file(MANIFEST),
        "survey_protocol_sha256": _sha_file(SURVEY),
        "n_survey_records": len(records),
        "sources_with_human_decisions": {
            k: {"n_event_logs": len(v), "paths": v} for k, v in sorted(reviewed.items())
        },
        "never_human_reviewed_sources": never_reviewed,
        "hard_exclude_sources": sorted(hard_exclude),
        "hard_exclude_reasons": {
            "008_sources": sorted(run1_sources),
            "presentation_repair_smoke_sources": sorted(smoke_sources),
            "historical_CE": ["MV-DVID-RAW-SURVEY-017", "MV-DVID-RAW-SURVEY-027"],
            "any_nonempty_event_log": sorted(reviewed),
        },
        "eligible_independent_never_reviewed": eligible_independent,
        "per_source_hashes": {
            sid: {
                "raw_sha256": records[sid]["raw_sha256"],
                "raw_path": records[sid]["raw_path"],
                "shape_zyx": records[sid]["shape_zyx"],
            }
            for sid in eligible_independent
        },
        "smoke_centers_excluded": [
            {"source_id": s, "center_zyx": list(c)} for s, c in sorted(smoke_centers)
        ],
        "run1_008_centers_excluded": [
            {"source_id": s, "center_zyx": list(c)} for s, c in sorted(run1_centers)
        ],
        "independence_requirement": {
            "n_locations_required": N_LOCATIONS,
            "prefer_one_location_per_independent_source": True,
            "available_independent_sources": len(eligible_independent),
            "satisfied": len(eligible_independent) >= N_LOCATIONS,
        },
    }
    INVENTORY.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ui_hash = _sha_file(PRES_MOD)
    reviewer_hash = _sha_file(REVIEW_BATCH)
    gate_hash = _sha_file(GATE_OUT)

    independence_ok = len(eligible_independent) >= N_LOCATIONS
    status = (
        "FROZEN_READY_TO_PROVISION"
        if independence_ok
        else "STOPPED_FRESH_SOURCE_INDEPENDENCE_UNSATISFIED"
    )

    protocol = {
        "id": "AFFINITY_SPEC002_VALIDATION_PROTOCOL_001",
        "schema_version": 1,
        "batch_id": BATCH_ID,
        "created_at": _now(),
        "status": status,
        "frozen_before_human_labels": True,
        "not_g3_009": True,
        "not_a_g3_training_cohort": True,
        "do_not_train": True,
        "smoke_labels_are_not_production_gt": True,
        "target_spec": "MV-G3-AFFINITY-TARGET-SPEC-002",
        "scientific_question": (
            "Can the repaired, presentation-equivariant, axis-blinded annotation procedure "
            "recover BOTH biological affinity classes on fresh isotropic FIB-SEM tissue "
            "while remaining orientation-invariant?"
        ),
        "annotation_question_wording": (
            "Do the two marked locations belong to the same biological process/object across their shared face?"
        ),
        "presentation": {
            "mode": "EQUIVARIANT_EDGE_FACE_V1",
            "module_path": "src/mvconnectome/equivariant_edge_presentation.py",
            "module_sha256": ui_hash,
            "reviewer_path": "tools/review_equivariant_edge_batch.py",
            "reviewer_sha256": reviewer_hash,
            "requirements": [
                "synthetic_permutation_equivariance_pass",
                "both_endpoints_equivalently_visible",
                "identical_physical_FOV_nm",
                "identical_normalization",
                "identical_marker_semantics",
                "no_axis_specific_context_construction",
                "no_axis_labels_during_judgment",
            ],
        },
        "synthetic_gate": {
            "path": str(GATE_OUT.relative_to(REPO)).replace("\\", "/"),
            "sha256": gate_hash,
            "overall_pass": True,
            "fixtures_sha256": gate.get("fixtures_sha256"),
            "decision": gate.get("decision"),
        },
        "design": {
            "n_locations": N_LOCATIONS,
            "edges_per_location": 3,
            "total_decisions": N_LOCATIONS * 3,
            "one_location_per_independent_source": True,
            "raw_only_strata": ["Q1", "Q2", "Q3", "Q4"],
            "strata_definition": (
                "Relative quartiles of mean |single-voxel intensity step| over the three "
                "+1 faces at each candidate center, pooled within the eligible independent "
                "fresh-source population. NOT historical C/E absolute thresholds. "
                "Strata are NOT SAME-like or DIFFERENT-like."
            ),
            "selection_inputs_allowed": [
                "coordinates",
                "raw_intensity",
                "local_contrast",
                "gradient_magnitude",
                "raw_pixel_membrane_likeness",
                "geometric_validity",
                "spatial_separation",
            ],
            "selection_inputs_forbidden": [
                "human_labels",
                "invalid_g3_labels",
                "smoke_labels",
                "model_affinity_predictions",
                "segmentation_predictions",
                "targeting_inferred_SAME_DIFFERENT",
            ],
            "seed_material": SEED,
        },
        "blinding": {
            "axis_blind": True,
            "opaque_decision_ids": True,
            "deterministic_pseudorandom_order": True,
            "reveal_axis_only_after_all_decisions_frozen": True,
            "do_not_reveal_during_judgment": [
                "X/Y/Z identity",
                "fixed axis order",
                "raw stratum",
                "historical labels",
                "expected outcome",
            ],
        },
        "predeclared_verdict_rules": {
            "NEGATIVE_CLASS_NOT_DEMONSTRATED": (
                "Zero DIFFERENT_PROCESS decisions across the entire validation batch "
                "(after excluding UNCERTAIN/BAD_QUESTION from class counts)."
            ),
            "ORIENTATION_INVARIANCE_FAILED": (
                "After unblinding, class is pathologically determined by global axis: "
                "every evaluable location triplet matches Z=DIFFERENT_PROCESS & "
                "Y=SAME_PROCESS & X=SAME_PROCESS (the historical orientation-collinear pattern), "
                "OR any single axis is perfectly predictive of class across all evaluable edges "
                "(one axis all DIFFERENT and never SAME while another axis all SAME and never DIFFERENT, "
                "with ≥3 evaluable edges on each of those axes)."
            ),
            "SPEC002_ANNOTATION_PROCEDURE_VALIDATED": (
                "Synthetic gate PASS; provenance valid; complete blinded review; "
                "≥1 DIFFERENT_PROCESS; ≥1 SAME_PROCESS; "
                "NOT NEGATIVE_CLASS_NOT_DEMONSTRATED; NOT ORIENTATION_INVARIANCE_FAILED; "
                "no presentation-leakage evidence sufficient to explain labels."
            ),
            "SPEC002_PROCEDURE_UNRESOLVED": (
                "Any other semantic/presentation integrity failure, incomplete review, "
                "or ambiguous outcome not covered above."
            ),
            "priority_order": [
                "NEGATIVE_CLASS_NOT_DEMONSTRATED",
                "ORIENTATION_INVARIANCE_FAILED",
                "SPEC002_PROCEDURE_UNRESOLVED",
                "SPEC002_ANNOTATION_PROCEDURE_VALIDATED",
            ],
            "exclusions_from_class_counts": ["UNCERTAIN", "BAD_QUESTION"],
            "no_threshold_tuning_after_labels": True,
            "validation_labels_are_not_automatic_production_gt": True,
        },
        "source_inventory_path": str(INVENTORY.relative_to(REPO)).replace("\\", "/"),
        "independence_check": inventory["independence_requirement"],
        "stop_reason": None
        if independence_ok
        else (
            f"Only {len(eligible_independent)} never-human-reviewed independent survey sources "
            f"available {eligible_independent}; design requires {N_LOCATIONS} with one location "
            f"per independent source. Refusing to weaken independence by stacking locations on "
            f"already-reviewed crops or by reusing -008/smoke sources."
        ),
        "next_required_if_stopped": (
            None
            if independence_ok
            else (
                "Acquire ≥12 new provenance-clean DVID survey crops that have never received "
                "human boundary decisions, then re-run this protocol without changing verdict rules."
            )
        ),
    }
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "protocol": str(PROTOCOL),
                "status": status,
                "synthetic_gate": True,
                "independent_sources": eligible_independent,
                "n_independent": len(eligible_independent),
                "required": N_LOCATIONS,
            },
            indent=2,
        )
    )
    return 0 if independence_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
