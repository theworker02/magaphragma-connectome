"""Reassess SPEC002 validation independence after new survey crops are acquired.

Updates inventory + protocol independence/status fields only.
Does NOT alter predeclared_verdict_rules.
Does NOT provision packages.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_PROTOCOL_001.json"
INVENTORY = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_SOURCE_INVENTORY_001.json"
MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
SURVEY = REPO / "experiments/phase6e/MV-G3-NEW-REGION-SURVEY-001.json"
SMOKE = REPO / "experiments/phase6e/G3_PRESENTATION_REPAIR_SMOKE_001.json"
RUN1 = REPO / "bin/run1.json"
ACQUISITION = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_SOURCE_ACQUISITION_001.json"
AMENDMENT = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_PROTOCOL_001_AMENDMENT_SOURCE_ACQUISITION.json"

N_LOCATIONS = 12


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources_with_human_decisions() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for root in (REPO / "experiments/phase6e", REPO / "local_research_build/phase6e"):
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
    if not PROTOCOL.exists():
        raise SystemExit("Protocol missing; run freeze_affinity_spec002_validation_protocol.py first")
    if not ACQUISITION.exists():
        raise SystemExit("Acquisition receipt missing; run acquire_spec002_validation_survey_crops.py first")
    if AMENDMENT.exists():
        raise SystemExit(f"Amendment already exists: {AMENDMENT}")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") not in {
        "STOPPED_FRESH_SOURCE_INDEPENDENCE_UNSATISFIED",
        "FROZEN_READY_TO_PROVISION",
    }:
        raise SystemExit(f"Unexpected protocol status: {protocol.get('status')}")

    # Guard: verdict rules must remain byte-identical to the frozen copy we keep.
    frozen_rules = protocol["predeclared_verdict_rules"]

    acquisition = json.loads(ACQUISITION.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    survey = json.loads(SURVEY.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    run1 = json.loads(RUN1.read_text(encoding="utf-8"))
    reviewed = sources_with_human_decisions()

    smoke_sources = set(smoke["design"]["sources"])
    run1_sources = set(run1["fresh_sources"])
    hard_exclude = set(smoke_sources) | set(run1_sources) | {
        "MV-DVID-RAW-SURVEY-017",
        "MV-DVID-RAW-SURVEY-027",
    }

    records = {r["id"]: r for r in manifest["records"]}
    never_reviewed = sorted(sid for sid in records if sid not in reviewed)
    eligible_independent = [sid for sid in never_reviewed if sid not in hard_exclude]

    inventory = {
        "id": "AFFINITY_SPEC002_VALIDATION_SOURCE_INVENTORY_001",
        "created_at": _now(),
        "supersedes_prior_inventory_snapshot": True,
        "survey_manifest_sha256": _sha_file(MANIFEST),
        "survey_protocol_sha256": _sha_file(SURVEY),
        "acquisition_receipt_sha256": _sha_file(ACQUISITION),
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
        "newly_acquired_sources": acquisition["records"],
        "per_source_hashes": {
            sid: {
                "raw_sha256": records[sid]["raw_sha256"],
                "raw_path": records[sid]["raw_path"],
                "shape_zyx": records[sid]["shape_zyx"],
            }
            for sid in eligible_independent
        },
        "independence_requirement": {
            "n_locations_required": N_LOCATIONS,
            "prefer_one_location_per_independent_source": True,
            "available_independent_sources": len(eligible_independent),
            "satisfied": len(eligible_independent) >= N_LOCATIONS,
        },
    }
    INVENTORY.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    independence_ok = len(eligible_independent) >= N_LOCATIONS
    status = (
        "FROZEN_READY_TO_PROVISION"
        if independence_ok
        else "STOPPED_FRESH_SOURCE_INDEPENDENCE_UNSATISFIED"
    )

    protocol["status"] = status
    protocol["independence_check"] = inventory["independence_requirement"]
    protocol["source_inventory_path"] = str(INVENTORY.relative_to(REPO)).replace("\\", "/")
    protocol["stop_reason"] = (
        None
        if independence_ok
        else (
            f"Only {len(eligible_independent)} never-human-reviewed independent survey sources "
            f"available; design requires {N_LOCATIONS}."
        )
    )
    protocol["next_required_if_stopped"] = (
        None
        if independence_ok
        else (
            "Acquire ≥12 new provenance-clean DVID survey crops that have never received "
            "human boundary decisions, then re-run this protocol without changing verdict rules."
        )
    )
    protocol["reassessed_at"] = _now()
    protocol["reassessment"] = {
        "amendment_id": "AFFINITY_SPEC002_VALIDATION_PROTOCOL_001_AMENDMENT_SOURCE_ACQUISITION",
        "acquisition_receipt": str(ACQUISITION.relative_to(REPO)).replace("\\", "/"),
        "verdict_rules_unchanged": True,
        "verdict_rules_sha256": hashlib.sha256(
            json.dumps(frozen_rules, sort_keys=True).encode()
        ).hexdigest(),
    }
    # Hard assert: rules object identity preserved
    if protocol["predeclared_verdict_rules"] != frozen_rules:
        raise SystemExit("INTERNAL: verdict rules mutated")

    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    amendment = {
        "id": "AFFINITY_SPEC002_VALIDATION_PROTOCOL_001_AMENDMENT_SOURCE_ACQUISITION",
        "created_at": _now(),
        "parent_protocol": "AFFINITY_SPEC002_VALIDATION_PROTOCOL_001",
        "purpose": "Record source acquisition and independence reassessment; do not alter verdict rules.",
        "status_before": "STOPPED_FRESH_SOURCE_INDEPENDENCE_UNSATISFIED",
        "status_after": status,
        "n_independent_before": 2,
        "n_independent_after": len(eligible_independent),
        "newly_acquired_ids": [r["id"] for r in acquisition["records"]],
        "verdict_rules_unchanged": True,
        "verdict_rules_sha256": protocol["reassessment"]["verdict_rules_sha256"],
        "allowed_mutations": [
            "status",
            "independence_check",
            "stop_reason",
            "next_required_if_stopped",
            "source_inventory contents",
            "reassessed_at / reassessment metadata",
        ],
        "forbidden_mutations": [
            "predeclared_verdict_rules",
            "design.n_locations",
            "design.total_decisions",
            "blinding",
            "scientific_question",
        ],
    }
    AMENDMENT.write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": status,
                "n_independent": len(eligible_independent),
                "required": N_LOCATIONS,
                "eligible": eligible_independent,
                "amendment": str(AMENDMENT),
            },
            indent=2,
        )
    )
    return 0 if independence_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
