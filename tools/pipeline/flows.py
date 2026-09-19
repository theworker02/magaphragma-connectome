"""Deterministic multi-tool pipelines (geometry/graph/stats first)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "receipts" / "pipeline"


def _write(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    return path


def plan_pipeline(
    *,
    shape_zyx: tuple[int, int, int] = (64, 64, 64),
    tile_zyx: tuple[int, int, int] = (32, 32, 32),
    volume: np.ndarray | None = None,
    region_id: str | None = None,
) -> dict[str, Any]:
    """VoxScout analyze -> AxonForge optimize_ok hints (+ optional RegionBench volume)."""
    from voxscout.scout import VoxScout
    from _core.artifactvet import ArtifactVet

    if volume is None and region_id:
        from regionbench.bench import load_volume, ensure_frozen
        ensure_frozen()
        volume = load_volume(region_id)
        shape_zyx = tuple(int(x) for x in volume.shape)
    if volume is None:
        # deterministic synthetic volume
        rng = np.random.default_rng(7)
        volume = rng.integers(0, 200, size=shape_zyx, dtype=np.uint8)
        volume[shape_zyx[0]//4:3*shape_zyx[0]//4,
               shape_zyx[1]//4:3*shape_zyx[1]//4,
               shape_zyx[2]//4:3*shape_zyx[2]//4] = 180

    pm = VoxScout(tile_zyx).analyze(volume)
    hints = pm.as_axonforge_hints()
    map_path = REPO / "receipts" / "voxscout" / "priority_map.json"
    pm.save(map_path)
    hints_path = OUT / "axonforge_hints.json"
    OUT.mkdir(parents=True, exist_ok=True)
    hints_path.write_text(json.dumps(hints, indent=2, sort_keys=True) + chr(10), encoding="utf-8")

    vet = ArtifactVet().vet_json(map_path, required_keys=("volume_shape", "tiles", "n_tiles"))
    result = {
        "pipeline": "plan",
        "steps": [
            {"tool": "voxscout", "action": "analyze", "n_tiles": len(pm.tiles), "class_counts": pm.class_counts()},
            {"tool": "axonforge", "action": "hints", "n_hints": len(hints), "optimize_ok": sum(1 for h in hints.values() if h.get("optimize_ok"))},
        ],
        "priority_map": str(map_path),
        "hints": str(hints_path),
        "artifactvet": vet.as_dict(),
        "region_id": region_id,
        "shape_zyx": list(shape_zyx),
        "tile_zyx": list(tile_zyx),
        "note": "Hints only — never discards tiles; optimize_ok guides AxonForge scheduling.",
    }
    result["receipt"] = str(_write("plan.json", result))
    return result


def close_gaps_pipeline(
    *,
    shape_zyx: tuple[int, int, int] = (100, 100, 100),
    run_tilemedic: bool = True,
    max_recoveries: int = 8,
) -> dict[str, Any]:
    """GapHound scan -> repair-plan -> optional TileMedic on failed items."""
    from gaphound.scan import GapHound
    from tilemedic.medic import TileMedic
    from _core.artifactvet import vet_gaphound_scan

    gh = GapHound()
    report = gh.scan(shape_zyx)
    plan = gh.repair_plan(report)
    scan_path = REPO / "receipts" / "gaphound" / "latest_scan.json"
    vet = vet_gaphound_scan(scan_path)

    recoveries = []
    if run_tilemedic:
        medic = TileMedic()
        failed = [w for w in plan.get("worklist", []) if w.get("kind") == "failed"][:max_recoveries]
        for item in failed:
            failure = {
                "tile_key": item.get("tile_key"),
                "error": item.get("reason") or "oom",
                "code": "OOM" if "oom" in str(item.get("reason", "")).lower() else "FAILED",
            }
            receipt = medic.recover(failure, config={"batch_size": 8})
            recoveries.append(receipt.as_dict())

    result = {
        "pipeline": "close-gaps",
        "steps": [
            {"tool": "gaphound", "action": "scan", "counts": report.counts, "coverage": report.coverage},
            {"tool": "gaphound", "action": "repair-plan", "n_work_items": plan["n_work_items"]},
            {"tool": "tilemedic", "action": "recover", "n_recoveries": len(recoveries), "skipped": not run_tilemedic},
        ],
        "repair_plan": plan,
        "recoveries": recoveries,
        "artifactvet": vet.as_dict(),
        "shape_zyx": list(shape_zyx),
    }
    result["receipt"] = str(_write("close_gaps.json", result))
    return result



def reconcile_pipeline() -> dict[str, Any]:
    """SeamSmith REVIEW candidates -> BranchJudge evidence assist."""
    from seamsmith.branch_assist import assist_review_candidates

    candidates = [
        {
            "recommended_action": "REVIEW",
            "evidence": {
                "spatial_distance": 3.2,
                "confidence": 0.42,
                "morphology_agreement": 0.55,
                "direction_agreement": 0.48,
            },
            "a": {"label_id": 1},
            "b": {"label_id": 2},
        },
        {
            "recommended_action": "KEEP_SPLIT",
            "evidence": {"spatial_distance": 12.0, "confidence": 0.9},
            "a": {"label_id": 3},
            "b": {"label_id": 4},
        },
    ]
    assisted = assist_review_candidates(candidates)
    result = {
        "pipeline": "reconcile",
        "steps": [
            {"tool": "seamsmith", "action": "review-candidates", "n": len(candidates)},
            {"tool": "branchjudge", "action": "assist", "n_assisted": len(assisted)},
        ],
        "assisted": assisted,
        "note": "Evidence assist only — never auto-merges.",
    }
    result["receipt"] = str(_write("reconcile.json", result))
    return result


def audit_pipeline(*, seed: int = 0) -> dict[str, Any]:
    """MorphGuard synthetic morphology audit."""
    import numpy as np
    from morphguard.audit import MorphGuard

    rng = np.random.default_rng(seed)
    labels = np.zeros((32, 32, 32), dtype=np.int32)
    labels[4:20, 4:20, 4:20] = 1
    labels[16:28, 16:28, 16:28] = 2
    labels[8, 8, 8] = 3  # tiny fragment
    _ = rng  # deterministic seed reserved
    amap = MorphGuard().analyze(labels=labels)
    result = {
        "pipeline": "audit",
        "steps": [
            {
                "tool": "morphguard",
                "action": "analyze",
                "n_anomalies": len(amap.anomalies),
                "kind_counts": amap.kind_counts(),
                "review_queue": len(amap.review_queue()),
            }
        ],
        "note": "Heuristic morphology flags only — not definitive biology errors.",
    }
    result["receipt"] = str(_write("audit.json", result))
    return result


def connectivity_pipeline() -> dict[str, Any]:
    """SynapseLens classify + prioritize on synthetic candidates."""
    from synapselens.evidence import CandidateSynapse, build_report

    cands = [
        CandidateSynapse(
            pre_id="p1", post_id="q1", xyz=(1.0, 2.0, 3.0),
            prediction_confidence=0.4, local_image_evidence=0.35,
            segmentation_confidence=0.4, boundary_status="uncertain",
            morphological_context=0.3, model_agreement=0.3,
        ),
        CandidateSynapse(
            pre_id="p2", post_id="q2", xyz=(5.0, 5.0, 5.0),
            prediction_confidence=0.91, local_image_evidence=0.88,
            segmentation_confidence=0.9, boundary_status="clean",
            morphological_context=0.85, model_agreement=0.92,
        ),
    ]
    report = build_report(cands)
    result = {
        "pipeline": "connectivity",
        "steps": [
            {
                "tool": "synapselens",
                "action": "prioritize",
                "total": report.total,
                "counts": dict(report.counts),
                "review_queue": len(report.review_queue),
            }
        ],
    }
    result["receipt"] = str(_write("connectivity.json", result))
    return result


def diff_pipeline() -> dict[str, Any]:
    """DeltaGraph structural diff between two synthetic builds."""
    from deltagraph.diff import Build, DeltaGraph
    from deltagraph.causes import infer

    a = Build(
        neurons={"n1": {"centroid": [0.0, 0.0, 0.0]}},
        synapses={}, edges=[("n1", "n2")], components=[["n1"]], degrees={"n1": 1},
        meta={"build": "A"},
    )
    b = Build(
        neurons={
            "n1": {"centroid": [0.0, 0.0, 0.0]},
            "n2": {"centroid": [3.0, 0.0, 0.0]},
        },
        synapses={"s1": {"confidence": 0.8}},
        edges=[("n1", "n2")], components=[["n1", "n2"]], degrees={"n1": 1, "n2": 1},
        meta={"build": "B"},
    )
    delta = DeltaGraph.diff(a, b)
    d = delta.to_dict()
    causes = infer(delta, a.meta, b.meta)
    result = {
        "pipeline": "diff",
        "steps": [
            {
                "tool": "deltagraph",
                "action": "diff",
                "neurons_added": len(d.get("neurons_added", [])),
                "synapses_added": len(d.get("synapses_added", [])),
            },
            {"tool": "deltagraph", "action": "infer-causes", "n_causes": len(causes) if isinstance(causes, list) else 1},
        ],
        "delta": d,
        "causes": causes if isinstance(causes, (dict, list)) else str(causes),
    }
    result["receipt"] = str(_write("diff.json", result))
    return result


def mass_status_pipeline() -> dict[str, Any]:
    """HyperDrain fleet / mass-production status into project receipts."""
    from hyperdrain.fleet import project_fleet
    try:
        from hyperdrain.queue import status_summary
        queue = status_summary()
    except Exception as exc:  # noqa: BLE001
        queue = {"error": str(exc)}
    fleet = project_fleet(1)
    result = {
        "pipeline": "mass-status",
        "steps": [
            {"tool": "hyperdrain", "action": "fleet", "n_gpus": fleet.get("n_gpus")},
            {"tool": "hyperdrain", "action": "queue-status"},
        ],
        "fleet": fleet,
        "queue": queue,
    }
    result["receipt"] = str(_write("mass_status.json", result))
    return result

