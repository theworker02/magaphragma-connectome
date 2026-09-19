"""Provision AFFINITY_SPEC002_VALIDATION_002 — abs-enriched dual-class check.

Changes vs AS2-001 (which failed NEGATIVE_CLASS_NOT_DEMONSTRATED):
  * Sources are new never-reviewed crops near C/E-magnitude anchors
  * Location centers require max single-face |ΔI| ≥ ABS_MIN (not relative Q1–Q4)
  * Still emits all three axes per location for orientation-invariance test
  * Same predeclared verdict rules as PROTOCOL_001; not -009; do not train
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
MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
GATE = REPO / "experiments/phase6e/G3_PRESENTATION_EQUIVARIANCE_REPAIR_001.json"
ACQ = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_002_SOURCE_ACQUISITION.json"
WS_ROOT = REPO / "experiments/phase6e/affinity-spec002-validation-002-packages"
Q_ROOT = REPO / "experiments/phase6e/affinity-spec002-validation-002-queues"
BATCH_OUT = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_002.json"
SEALED = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_002_AXIS_MAP_SEALED.json"
INVENTORY = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_002_SOURCE_INVENTORY.json"

N_LOCATIONS = 12
ABS_MIN = 12.0
SEED = "AFFINITY_SPEC002_VALIDATION_002|ABS_FACE_ENRICHED|EQUIVARIANT_EDGE_FACE_V1"
BATCH_ID = "AFFINITY_SPEC002_VALIDATION_002"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _hash_rank(*parts: str) -> int:
    return int(_sha(*parts), 16)


def opaque_id(seq: int) -> str:
    return f"MV-AS2B-{_sha(BATCH_ID, f'{seq:04d}')[:12].upper()}"


def face_steps(arr: np.ndarray, center: tuple[int, int, int]) -> dict[str, float]:
    z, y, x = center
    a = np.asarray(arr, dtype=np.float64)
    return {
        "Z": float(abs(a[z + 1, y, x] - a[z, y, x])),
        "Y": float(abs(a[z, y + 1, x] - a[z, y, x])),
        "X": float(abs(a[z, y, x + 1] - a[z, y, x])),
    }


def sources_with_human_decisions() -> set[str]:
    found: set[str] = set()
    for root in (REPO / "experiments/phase6e", REPO / "local_research_build/phase6e"):
        if not root.exists():
            continue
        for log in root.rglob("workspace.events.jsonl"):
            if not log.read_text(encoding="utf-8").strip():
                continue
            ws_path = log.parent / "workspace.json"
            if not ws_path.exists():
                continue
            sid = json.loads(ws_path.read_text(encoding="utf-8")).get("parent_region_id")
            if sid:
                found.add(sid)
    return found


def select_locations(sources: list[str], records: dict) -> list[dict]:
    hist_list, _ = g8.pinned_historical_edges()
    hist_set = {(tuple(l), tuple(r), c) for (l, r, c) in hist_list}
    selection = []
    for sid in sources:
        arr = np.load(records[sid]["raw_path"], mmap_mode="r", allow_pickle=False)
        centers = g8.greedy_separated_centers(*g8.eligible_centers(arr))
        raw_sha = records[sid]["raw_sha256"]
        cands = []
        for (c, mean3) in centers:
            steps = face_steps(arr, c)
            mx = max(steps.values())
            if mx < ABS_MIN:
                continue
            edges = g8._edges(c)
            if any(
                (tuple(edges[ax][0]), tuple(edges[ax][1]), g8._axis_index(ax)) in hist_set
                for ax in ("Z", "Y", "X")
            ):
                continue
            cands.append((c, mean3, steps, mx))
        if not cands:
            raise SystemExit(f"{sid}: no abs>={ABS_MIN} center after exclusion")
        pick = min(cands, key=lambda cc: _hash_rank(SEED, raw_sha, sid, str(cc[0])))
        selection.append(
            {
                "source_id": sid,
                "center_zyx": list(pick[0]),
                "raw_contrast_mean3": pick[1],
                "face_steps": pick[2],
                "max_face_step": pick[3],
                "raw_sha256": raw_sha,
                "enrichment": "ABS_MAX_FACE_GE_12",
            }
        )
    return selection


def build() -> dict:
    if any(p.exists() for p in (WS_ROOT, Q_ROOT, BATCH_OUT, SEALED, INVENTORY)):
        raise FileExistsError("Refusing to overwrite AS2-002 artifacts")
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    if not gate.get("overall_pass"):
        raise SystemExit("Synthetic gate failed")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    acq = json.loads(ACQ.read_text(encoding="utf-8"))
    sources = sorted(acq["selected_ids"])
    if len(sources) < N_LOCATIONS:
        raise SystemExit(f"Need {N_LOCATIONS} sources, have {len(sources)}")
    sources = sources[:N_LOCATIONS]

    reviewed = sources_with_human_decisions()
    leaked = [s for s in sources if s in reviewed]
    if leaked:
        raise SystemExit(f"Selected sources already human-reviewed: {leaked}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = {r["id"]: r for r in manifest["records"]}
    for sid in sources:
        if sid not in records:
            raise SystemExit(f"missing {sid}")

    selection = select_locations(sources, records)
    inventory = {
        "id": "AFFINITY_SPEC002_VALIDATION_002_SOURCE_INVENTORY",
        "created_at": _now(),
        "selected_sources": sources,
        "n": len(sources),
        "enrichment": f"max single-face |ΔI| >= {ABS_MIN}",
        "vs_as2_001": (
            "AS2-001 used relative Q1–Q4 on mean3 face-step (~0.3–3 abs) and got "
            "NEGATIVE_CLASS_NOT_DEMONSTRATED. AS2-002 uses C/E-neighborhood fresh "
            "crops and abs-max-face enrichment after DIFFERENT_CLASS_DISCOVERY_001 "
            "demonstrated DIFFERENT on abs≥12 edges under equivariant UI."
        ),
        "discovery_prior": "experiments/phase6e/DIFFERENT_CLASS_DISCOVERY_001.json",
        "do_not_train": True,
        "not_g3_009": True,
    }
    INVENTORY.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    decisions = []
    for loc_i, loc in enumerate(selection):
        for ax in ("Z", "Y", "X"):
            left, right = g8._edges(tuple(loc["center_zyx"]))[ax]
            decisions.append(
                {
                    "source_id": loc["source_id"],
                    "center_zyx": loc["center_zyx"],
                    "raw_sha256": loc["raw_sha256"],
                    "axis_name": ax,
                    "channel_zyx": g8._axis_index(ax),
                    "pair_left_zyx": list(left),
                    "pair_right_zyx": list(right),
                    "face_step": loc["face_steps"][ax],
                    "location_index": loc_i,
                    "max_face_step": loc["max_face_step"],
                }
            )

    n_dec = N_LOCATIONS * 3
    order = sorted(range(n_dec), key=lambda i: _sha(SEED, "ORDER", f"{i:02d}"))
    ordered = [decisions[i] for i in order]
    for seq, d in enumerate(ordered):
        d["opaque_decision_id"] = opaque_id(seq)
        d["presentation_order_index"] = seq

    order_hash = _sha(SEED, "ORDER_DIGEST", ",".join(d["opaque_decision_id"] for d in ordered))
    mapping_hash = _sha(
        SEED,
        "MAP",
        json.dumps([(d["opaque_decision_id"], d["axis_name"], d["center_zyx"]) for d in ordered], sort_keys=True),
    )

    WS_ROOT.mkdir(parents=True)
    Q_ROOT.mkdir(parents=True)
    sealed_entries = []
    per_location = []
    for loc_i, loc in enumerate(selection):
        crop_id = f"MV-AS2B-VAL-{loc_i + 1:02d}"
        sid = loc["source_id"]
        r = records[sid]
        crop_dir = WS_ROOT / crop_id
        crop_dir.mkdir(parents=True)
        log = crop_dir / "workspace.events.jsonl"
        log.write_text("", encoding="utf-8")
        ws_id = f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id}-AS2B"
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
                "enrichment": loc["enrichment"],
                "max_face_step": loc["max_face_step"],
            },
            "review_state": "UNREVIEWED",
            "schema_version": 1,
            "source_origin_xyz": [r["bounds_xyz"][a][0] for a in ("x", "y", "z")],
            "split": "AFFINITY_SPEC002_VALIDATION_002_ONLY",
            "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
        }
        (crop_dir / "workspace.json").write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        for d in ordered:
            if d["location_index"] != loc_i:
                continue
            sealed_entries.append(
                {
                    "opaque_decision_id": d["opaque_decision_id"],
                    "presentation_order_index": d["presentation_order_index"],
                    "crop_id": crop_id,
                    "source_id": sid,
                    "center_zyx": d["center_zyx"],
                    "axis_name": d["axis_name"],
                    "channel_zyx": d["channel_zyx"],
                    "pair_left_zyx": d["pair_left_zyx"],
                    "pair_right_zyx": d["pair_right_zyx"],
                    "face_step": d["face_step"],
                    "workspace_path": str((crop_dir / "workspace.json").resolve()),
                }
            )
        per_location.append(
            {
                "crop_id": crop_id,
                "source_id": sid,
                "center_zyx": loc["center_zyx"],
                "max_face_step": loc["max_face_step"],
                "face_steps": loc["face_steps"],
                "workspace_id": ws_id,
                "n_questions": 3,
            }
        )

    master_questions = []
    for d in ordered:
        crop_id = f"MV-AS2B-VAL-{d['location_index'] + 1:02d}"
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
    master_path = Q_ROOT / "MASTER_BATCH.json"
    master_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": f"{BATCH_ID}-MASTER-QUEUE",
                "validation_batch_id": BATCH_ID,
                "created_at": _now(),
                "status": "EXPERT_EQUIVARIANT_VALIDATION_REVIEW_REQUIRED",
                "axis_blind": True,
                "expected_n_decisions": n_dec,
                "questions": master_questions,
                "scientific_boundary": (
                    "AS2-002 abs-enriched SPEC002 validation. Not training. Not G3-009."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    SEALED.write_text(
        json.dumps(
            {
                "id": "AFFINITY_SPEC002_VALIDATION_002_AXIS_MAP_SEALED",
                "status": "SEALED_UNTIL_ALL_36_DECISIONS_COMPLETE",
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
        "protocol_id": "AFFINITY_SPEC002_VALIDATION_PROTOCOL_001",
        "supersedes_attempt": "AFFINITY_SPEC002_VALIDATION_001",
        "not_a_g3_training_cohort": True,
        "not_experiment_009": True,
        "do_not_train": True,
        "synthetic_gate": {"overall_pass": True, "path": str(GATE.relative_to(REPO)).replace("\\", "/")},
        "design": {
            "n_locations": N_LOCATIONS,
            "edges_per_location": 3,
            "total_decisions": n_dec,
            "sources": sources,
            "enrichment": f"max single-face |ΔI| >= {ABS_MIN}",
            "selection_rule": (
                "One hash-min abs-enriched center per fresh C/E-neighborhood crop; "
                "all three +1 faces reviewed; pinned historical edge exclusion."
            ),
            "why_not_relative_q": inventory["vs_as2_001"],
        },
        "blinding": {
            "axis_blind_during_judgment": True,
            "order_hash": order_hash,
            "mapping_hash": mapping_hash,
            "sealed_axis_map_path": str(SEALED.relative_to(REPO)).replace("\\", "/"),
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
            r'experiments\phase6e\affinity-spec002-validation-002-queues\MASTER_BATCH.json '
            r'--reviewer matth'
        ),
        "results": None,
        "verdict": None,
    }
    BATCH_OUT.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return batch


if __name__ == "__main__":
    b = build()
    steps = [loc["max_face_step"] for loc in b["per_location"]]
    print(
        json.dumps(
            {
                "id": b["id"],
                "sources": b["design"]["sources"],
                "max_face_steps": steps,
                "order_hash": b["blinding"]["order_hash"],
            },
            indent=2,
        )
    )
