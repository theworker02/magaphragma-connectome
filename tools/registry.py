"""Connectome Project tool registry."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE_PATH = REPO / "receipts" / "tools_registry.json"

@dataclass
class ToolEntry:
    id: str
    name: str
    subsystem: str
    summary: str
    entrypoint: str
    dashboard: str = ""
    powered_on: bool = True
    autonomous: bool = False
    stage: str = ""

DEFAULTS: list[ToolEntry] = [
    ToolEntry("voxscout", "VoxScout", "VoxScout", "Intelligent spatial planner — priority map before expensive compute.", "python tools/voxscout.py", stage="plan"),
    ToolEntry("axonforge", "AxonForge", "AxonForge", "Completeness-preserving adaptive tile acceleration + light switch.", "python tools/axonforge.py", dashboard="http://127.0.0.1:8741/", stage="execute"),
    ToolEntry("gaphound", "GapHound", "GapHound", "Completeness hunter — scan gaps + minimum AxonForge repair-plan.", "python tools/gaphound.py", stage="completeness"),
    ToolEntry("tilemedic", "TileMedic", "TileMedic", "Safe failed-tile recovery without mutating scientific parameters.", "python tools/tilemedic.py", stage="recover"),
    ToolEntry("branchjudge", "BranchJudge", "BranchJudge", "Split/merge evidence assistant (evidence scores, not bio probabilities).", "python tools/branchjudge.py", stage="decide"),
    ToolEntry("seamsmith", "SeamSmith", "SeamSmith", "Boundary reconciliation across chunk/tile seams (never auto-merges).", "python tools/seamsmith.py", stage="reconcile"),
    ToolEntry("morphguard", "MorphGuard", "MorphGuard", "Neuron morphology auditor — anomaly map + review queue.", "python tools/morphguard.py", stage="audit"),
    ToolEntry("neurocache", "NeuroCache", "NeuroCache", "Semantic scientific computation cache (identity-aware reuse).", "python tools/neurocache.py", stage="cache"),
    ToolEntry("synapselens", "SynapseLens", "SynapseLens", "Synaptic evidence engine with information-gain review priority.", "python tools/synapselens.py", stage="connectivity"),
    ToolEntry("tracewire", "TraceWire", "TraceWire", "Scientific provenance debugger (forward/back walks).", "python tools/tracewire.py", stage="provenance"),
    ToolEntry("deltagraph", "DeltaGraph", "DeltaGraph", "Connectome structural diff + pipeline cause inference.", "python tools/deltagraph.py", stage="diff"),
    ToolEntry("runddoctor", "RunDoctor", "RunDoctor", "One-command pipeline diagnosis.", "python tools/runddoctor.py", stage="diagnose"),
    ToolEntry("hyperdrain", "HyperDrain", "HyperDrain", "Packed-eager affinity inference engine (mass production / fleet).", "python tools/hyperdrain.py", stage="execute"),
    ToolEntry("regionbench", "RegionBench", "RegionBench", "Frozen synthetic benchmark regions for comparable metrics.", "python tools/regionbench.py", stage="bench"),
    ToolEntry("pipeline", "Pipeline", "Pipeline", "Cross-tool flows: plan (VoxScout->hints) and close-gaps.", "python tools/pipeline.py", stage="orchestrate"),
    ToolEntry("edgeprobe", "EdgeProbe", "EdgeProbe", "TraceWire edge diagnostic bundle under receipts/edgeprobe/.", "python tools/edgeprobe.py", stage="provenance"),
    ToolEntry("artifactvet", "ArtifactVet", "ArtifactVet", "Validate schema/shape/dtype/hash before stage handoff.", "python tools/artifactvet.py", stage="gate"),
]

def _load_raw() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}

def save_registry(entries: list[ToolEntry]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({e.id: asdict(e) for e in entries}, indent=2) + chr(10), encoding="utf-8")

def list_tools() -> list[ToolEntry]:
    raw = _load_raw()
    out = []
    for d in DEFAULTS:
        o = raw.get(d.id, {})
        out.append(ToolEntry(
            id=d.id, name=o.get("name", d.name), subsystem=o.get("subsystem", d.subsystem),
            summary=o.get("summary", d.summary), entrypoint=o.get("entrypoint", d.entrypoint),
            dashboard=o.get("dashboard", d.dashboard), powered_on=bool(o.get("powered_on", d.powered_on)),
            autonomous=bool(o.get("autonomous", d.autonomous)), stage=o.get("stage", d.stage),
        ))
    return out

def get_tool(tool_id: str) -> ToolEntry | None:
    for t in list_tools():
        if t.id == tool_id: return t
    return None

def set_power(tool_id: str, *, powered_on: bool | None = None, autonomous: bool | None = None) -> ToolEntry:
    tools = list_tools()
    found = None
    for t in tools:
        if t.id == tool_id:
            if powered_on is not None: t.powered_on = powered_on
            if autonomous is not None: t.autonomous = autonomous
            found = t
            break
    if found is None: raise KeyError(tool_id)
    save_registry(tools)
    return found

def as_dict_list() -> list[dict]:
    return [asdict(t) for t in list_tools()]