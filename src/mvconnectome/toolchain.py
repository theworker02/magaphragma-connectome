"""Connectome project entry for the engineering toolchain.

Exposes every registered tool to the main project (`vigilia toolchain` / `connectome`) and documents how AxonForge + pipelines consume them.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


def ensure_tool_path() -> None:
    for p in (str(ROOT), str(TOOLS)):
        if p not in sys.path:
            sys.path.insert(0, p)


# How each registered tool is connected into the Connectome project.
CONNECTIONS: dict[str, dict[str, str]] = {
    "voxscout": {
        "via": "axonforge.bridge.apply_hints_to_graph + pipeline plan",
        "used_by": "AxonForge submit_volume / POST /pipeline/plan",
    },
    "axonforge": {
        "via": "axonforge.runtime + tools/axonforge.py + light switch",
        "used_by": "dashboard :8741 / vigilia toolchain switch",
    },
    "gaphound": {
        "via": "axonforge.bridge.post_run_gaphound + pipeline close-gaps",
        "used_by": "AxonForge request_inference post-run",
    },
    "tilemedic": {
        "via": "pipeline close-gaps recoveries",
        "used_by": "GapHound repair-plan failed tiles",
    },
    "branchjudge": {
        "via": "seamsmith.branch_assist + pipeline reconcile",
        "used_by": "SeamSmith REVIEW candidates",
    },
    "seamsmith": {
        "via": "pipeline reconcile + post_run_toolchain",
        "used_by": "AxonForge post-run boundary pass",
    },
    "morphguard": {
        "via": "pipeline audit + post_run_toolchain",
        "used_by": "AxonForge post-run morphology audit",
    },
    "neurocache": {
        "via": "axonforge.adaptive neurocache_lookup/store",
        "used_by": "AxonForge tile inference identity cache",
    },
    "synapselens": {
        "via": "pipeline connectivity + post_run_toolchain",
        "used_by": "AxonForge post-run synapse evidence",
    },
    "tracewire": {
        "via": "axonforge.bridge.record_trace_run + edgeprobe",
        "used_by": "AxonForge request_inference provenance",
    },
    "deltagraph": {
        "via": "pipeline diff + post_run_toolchain",
        "used_by": "build-A vs build-B structural diff",
    },
    "runddoctor": {
        "via": "connectome doctor + post_run_toolchain",
        "used_by": "project health / import checks",
    },
    "hyperdrain": {
        "via": "pipeline mass-status + post_run_toolchain fleet",
        "used_by": "affinity mass-production status in project",
    },
    "regionbench": {
        "via": "pipeline plan --region + regionbench smoke",
        "used_by": "frozen synthetic volumes for VoxScout/AxonForge",
    },
    "pipeline": {
        "via": "tools/pipeline/flows.py + AxonForge API",
        "used_by": "cross-tool orchestration",
    },
    "edgeprobe": {
        "via": "TraceWire edge diagnostic bundles",
        "used_by": "connectome edgeprobe / post_run_toolchain",
    },
    "artifactvet": {
        "via": "plan/close-gaps gates + post_run receipt vet",
        "used_by": "stage handoff validation",
    },
}


def connection_status() -> dict[str, Any]:
    """Import every registered tool and report project connection map."""
    ensure_tool_path()
    import registry

    tools = registry.list_tools()
    importable: dict[str, bool] = {}
    errors: dict[str, str] = {}
    for t in tools:
        mod = t.id if t.id != "artifactvet" else "_core.artifactvet"
        try:
            if t.id == "axonforge":
                __import__("axonforge")
            elif t.id == "artifactvet":
                __import__("_core.artifactvet")
            else:
                __import__(t.id)
            importable[t.id] = True
        except Exception as exc:  # noqa: BLE001
            importable[t.id] = False
            errors[t.id] = str(exc)

    connected = {tid: CONNECTIONS.get(tid, {"via": "registered only", "used_by": "CLI"}) for tid in importable}
    missing = [tid for tid, ok in importable.items() if not ok]
    unwired = [tid for tid in importable if tid not in CONNECTIONS]
    return {
        "project": "vigilia-connectome",
        "n_tools": len(tools),
        "n_importable": sum(1 for v in importable.values() if v),
        "n_connected": len(CONNECTIONS),
        "importable": importable,
        "errors": errors,
        "connections": connected,
        "missing_imports": missing,
        "unwired": unwired,
        "ok": not missing and not unwired,
    }


def main(argv: list[str] | None = None) -> int:
    """Forward to tools/connectome.py, with a project-level status command."""
    ensure_tool_path()
    args = list(argv) if argv is not None else sys.argv[1:]
    if args and args[0] == "status":
        import json
        print(json.dumps(connection_status(), indent=2, sort_keys=True))
        return 0 if connection_status()["ok"] else 1
    import connectome as connectome_cli
    return int(connectome_cli.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
