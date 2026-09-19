"""Bridge: Connectome toolchain <-> AxonForge runtime.

Loads VoxScout hints, NeuroCache tile results, TraceWire provenance, and
GapHound post-run scans so tools are actually consumed by the project.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
HINTS_PATH = REPO / "receipts" / "pipeline" / "axonforge_hints.json"
PRIORITY_PATH = REPO / "receipts" / "voxscout" / "priority_map.json"

for p in (str(REPO), str(TOOLS)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _af_grid_key(z: int, y: int, x: int) -> str:
    return f"T-Z{z:03d}-Y{y:03d}-X{x:03d}"


def tile_task_scout_key(task) -> str:
    """Map AxonForge TileTask / AF-VOLUME key to VoxScout T-Z### style."""
    tile = getattr(task, "tile", None)
    if tile is not None:
        return _af_grid_key(int(tile.z), int(tile.y), int(tile.x))
    key = getattr(task, "key", None) or ""
    # AF-VOLUME-X000-Y000-Z000-L0
    if "X" in key and "Y" in key and "Z" in key:
        try:
            parts = key.replace("AF-VOLUME-", "").split("-")
            xd = {p[0]: int(p[1:]) for p in parts if p and p[0] in "XYZL"}
            return _af_grid_key(xd.get("Z", 0), xd.get("Y", 0), xd.get("X", 0))
        except Exception:
            return key
    return str(key)


def load_hints(path: Path | None = None) -> dict[str, dict]:
    path = path or HINTS_PATH
    if not path.exists() and PRIORITY_PATH.exists():
        # derive from priority map
        try:
            from voxscout.priority_map import PriorityMap
            raw = json.loads(PRIORITY_PATH.read_text(encoding="utf-8"))
            # minimal reconstruct
            hints = {}
            for t in raw.get("tiles", []):
                k = t.get("tile_key")
                if not k:
                    continue
                hints[k] = {
                    "priority": t.get("priority", t.get("recommend_priority", 0.5)),
                    "class": t.get("region_class", t.get("class", "NORMAL")),
                    "optimize_ok": bool(t.get("optimize_ok", False)),
                    "box": t.get("box", {}),
                }
            return hints
        except Exception:
            return {}
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def apply_hints_to_graph(graph, hints: dict[str, dict] | None = None) -> dict[str, Any]:
    """Set task.priority from VoxScout hints (higher = sooner). Never drops tiles."""
    hints = hints if hints is not None else load_hints()
    applied = 0
    optimize_ok = 0
    for key, task in graph.tasks.items():
        sk = tile_task_scout_key(task)
        h = hints.get(sk) or hints.get(key)
        if not h:
            continue
        task.priority = float(h.get("priority", 0.0))
        if h.get("optimize_ok"):
            optimize_ok += 1
            # annotate reason; status unchanged — completeness preserved
            if not task.reason:
                task.reason = f"voxscout:{h.get('class', 'OPTIMIZE')}"
        applied += 1
    return {"hints_loaded": len(hints), "tasks_annotated": applied, "optimize_ok": optimize_ok}


def pending_by_priority(graph) -> list:
    tasks = [t for t in graph.tasks.values() if t.status == "PENDING"]
    tasks.sort(key=lambda t: (-float(getattr(t, "priority", 0.0)), t.tile.key if hasattr(t.tile, "key") else ""))
    return tasks


def neurocache_lookup_or_none(tile_key: str, box: dict, volume_hash: str = "synthetic") -> dict | None:
    try:
        from neurocache.axonforge_id import tile_inference_identity, lookup_tile_result
        ident = tile_inference_identity(volume_hash=volume_hash, tile_key=tile_key, box=box or {})
        return lookup_tile_result(ident)
    except Exception:
        return None


def neurocache_store(tile_key: str, box: dict, payload: dict, volume_hash: str = "synthetic") -> str | None:
    try:
        from neurocache.axonforge_id import tile_inference_identity, put_tile_result
        ident = tile_inference_identity(volume_hash=volume_hash, tile_key=tile_key, box=box or {})
        put_tile_result(ident, payload)
        return ident.identity_hash()
    except Exception:
        return None


def record_trace_run(*, shape_zyx: tuple, tiles_done: int, pass_name: str) -> None:
    try:
        from tracewire.wire import TraceWire
        tw = TraceWire.load() if hasattr(TraceWire, "load") else TraceWire()
        vol_id = f"volume:af-{shape_zyx[0]}x{shape_zyx[1]}x{shape_zyx[2]}"
        run_id = f"inference:axonforge-{pass_name}"
        if hasattr(tw, "add_node"):
            tw.add_node(vol_id, "volume", shape=list(shape_zyx))
            tw.add_node(run_id, "inference", tiles_done=tiles_done, pass_name=pass_name)
            tw.link(vol_id, run_id, "inferred_by")
        elif hasattr(tw, "register_chain"):
            tw.register_chain([
                ("volume", vol_id),
                ("inference", run_id),
            ])
        if hasattr(tw, "save"):
            tw.save()
    except Exception:
        pass


def post_run_gaphound(shape_zyx: tuple[int, int, int]) -> dict[str, Any]:
    try:
        from gaphound.scan import GapHound
        report = GapHound().scan(shape_zyx)
        plan = GapHound().repair_plan(report)
        return {
            "coverage": report.coverage,
            "missing": report.counts.get("missing", 0),
            "failed": report.counts.get("failed", 0),
            "repair_items": plan.get("n_work_items", 0),
        }
    except Exception as exc:
        return {"error": str(exc)}


def post_run_toolchain(shape_zyx: tuple[int, int, int], *, tiles_done: int = 0) -> dict[str, Any]:
    """Run every project-connected tool once after AxonForge inference.

    Deterministic synthetic inputs only — never fabricates biological claims.
    Failures are recorded; they do not abort the inference report.
    """
    out: dict[str, Any] = {"shape_zyx": list(shape_zyx), "tiles_done": tiles_done}
    out["gaphound"] = post_run_gaphound(shape_zyx)

    try:
        from runddoctor.doctor import RunDoctor
        doc = RunDoctor().run()
        out["runddoctor"] = {
            "n_checks": len(doc.checks),
            "n_ok": sum(1 for c in doc.checks if c.ok),
            "tool_imports": doc.tool_imports,
            "switch": doc.switch,
        }
    except Exception as exc:  # noqa: BLE001
        out["runddoctor"] = {"error": str(exc)}

    try:
        import numpy as np
        from morphguard.audit import MorphGuard
        labels = np.zeros((24, 24, 24), dtype=np.int32)
        labels[4:16, 4:16, 4:16] = 1
        labels[12:20, 12:20, 12:20] = 2
        amap = MorphGuard().analyze(labels=labels)
        out["morphguard"] = {
            "n_anomalies": len(amap.anomalies),
            "kind_counts": amap.kind_counts(),
            "review_queue": len(amap.review_queue()),
        }
    except Exception as exc:  # noqa: BLE001
        out["morphguard"] = {"error": str(exc)}

    try:
        from seamsmith.branch_assist import assist_review_candidates
        assisted = assist_review_candidates([
            {
                "recommended_action": "REVIEW",
                "evidence": {"spatial_distance": 3.5, "confidence": 0.4,
                             "morphology_agreement": 0.5, "direction_agreement": 0.5},
                "a": {"label_id": 1},
                "b": {"label_id": 2},
            }
        ])
        out["seamsmith"] = {"mode": "branch_assist", "n_review": 1}
        out["branchjudge"] = {"n_assisted": len(assisted)}
    except Exception as exc:  # noqa: BLE001
        out["seamsmith"] = {"error": str(exc)}
        out["branchjudge"] = {"error": str(exc)}

    try:
        from synapselens.evidence import CandidateSynapse, build_report
        cands = [
            CandidateSynapse(
                pre_id="syn-pre-1", post_id="syn-post-1", xyz=(10.0, 10.0, 10.0),
                prediction_confidence=0.55, local_image_evidence=0.5,
                segmentation_confidence=0.5, boundary_status="uncertain",
                morphological_context=0.45, model_agreement=0.4,
            ),
            CandidateSynapse(
                pre_id="syn-pre-2", post_id="syn-post-2", xyz=(20.0, 20.0, 20.0),
                prediction_confidence=0.92, local_image_evidence=0.88,
                segmentation_confidence=0.9, boundary_status="clean",
                morphological_context=0.85, model_agreement=0.9,
            ),
        ]
        report = build_report(cands)
        out["synapselens"] = {
            "total": report.total,
            "counts": dict(report.counts),
            "review_queue": len(report.review_queue),
        }
    except Exception as exc:  # noqa: BLE001
        out["synapselens"] = {"error": str(exc)}

    try:
        from deltagraph.diff import Build, DeltaGraph
        a = Build(
            neurons={"n1": {"centroid": [0.0, 0.0, 0.0]}},
            synapses={}, edges=[("n1", "n2")], components=[["n1"]], degrees={"n1": 1},
        )
        b = Build(
            neurons={
                "n1": {"centroid": [0.0, 0.0, 0.0]},
                "n2": {"centroid": [2.0, 0.0, 0.0]},
            },
            synapses={"s1": {"confidence": 0.7}},
            edges=[("n1", "n2")], components=[["n1", "n2"]], degrees={"n1": 1, "n2": 1},
        )
        delta = DeltaGraph.diff(a, b)
        d = delta.to_dict()
        out["deltagraph"] = {
            "neurons_added": len(d.get("neurons_added", [])),
            "synapses_added": len(d.get("synapses_added", [])),
            "edges_changed": {k: len(v) for k, v in (d.get("edges_changed") or {}).items()},
        }
    except Exception as exc:  # noqa: BLE001
        out["deltagraph"] = {"error": str(exc)}

    try:
        from edgeprobe.probe import edgeprobe
        out["edgeprobe"] = edgeprobe("af-pre", "af-post")
    except Exception as exc:  # noqa: BLE001
        out["edgeprobe"] = {"error": str(exc)}

    try:
        from hyperdrain.fleet import project_fleet
        fleet = project_fleet(1)
        out["hyperdrain"] = {
            "id": fleet.get("id"),
            "n_gpus": fleet.get("n_gpus"),
            "aggregate_tiles_per_sec": fleet.get("aggregate_tiles_per_sec"),
            "eta_hours_one_gpu": fleet.get("eta_hours_one_gpu"),
        }
    except Exception as exc:  # noqa: BLE001
        out["hyperdrain"] = {"error": str(exc)}

    try:
        from _core.artifactvet import ArtifactVet
        summary = REPO / "receipts" / "SUMMARY.json"
        if summary.exists():
            out["artifactvet"] = ArtifactVet().vet_json(
                summary, required_keys=("project", "subsystem")
            ).as_dict()
        else:
            out["artifactvet"] = {"ok": True, "skipped": "no SUMMARY.json yet"}
    except Exception as exc:  # noqa: BLE001
        out["artifactvet"] = {"error": str(exc)}

    out["tools_invoked"] = sorted(
        k for k in out if k not in ("shape_zyx", "tiles_done", "tools_invoked", "receipt")
    )
    try:
        dest = REPO / "receipts" / "pipeline" / "post_run_toolchain.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(out, indent=2, sort_keys=True, default=str) + chr(10),
            encoding="utf-8",
        )
        out["receipt"] = str(dest)
    except Exception:  # noqa: BLE001
        pass
    return out

