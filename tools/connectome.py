#!/usr/bin/env python3
"""Unified Connectome Project toolchain CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(TOOLS)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="connectome", description="Connectome Project engineering toolchain")
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("status", help="Project connection map for all tools")
    st.set_defaults(func=cmd_status)

    t = sub.add_parser("tools", help="List registered tools")
    t.set_defaults(func=cmd_tools)

    sw = sub.add_parser("switch", help="AxonForge light switch")
    sw.add_argument("action", choices=["on", "off", "toggle", "status", "autonomous", "manual"])
    sw.set_defaults(func=cmd_switch)

    gh = sub.add_parser("gaphound", help="Completeness hunter")
    gh_sub = gh.add_subparsers(dest="gh_cmd", required=True)
    ghs = gh_sub.add_parser("scan")
    ghs.add_argument("--shape", default="100,100,100")
    ghs.set_defaults(func=cmd_gaphound_scan)
    ghr = gh_sub.add_parser("repair-plan")
    ghr.set_defaults(func=cmd_gaphound_repair)

    tm = sub.add_parser("tilemedic", help="Failed-tile recovery")
    tm_sub = tm.add_subparsers(dest="tm_cmd", required=True)
    tmr = tm_sub.add_parser("recover")
    tmr.add_argument("--failed", action="store_true")
    tmr.add_argument("--error", default="oom")
    tmr.add_argument("--tile", default="T-Z000-Y000-X000")
    tmr.add_argument("--code", default="OOM")
    tmr.add_argument("--batch-size", type=int, default=8)
    tmr.set_defaults(func=cmd_tilemedic)

    bj = sub.add_parser("branchjudge", help="Split/merge evidence assistant")
    bj_sub = bj.add_subparsers(dest="bj_cmd", required=True)
    bjj = bj_sub.add_parser("judge")
    bjj.add_argument("--seg-confidence", type=float, default=0.45)
    bjj.add_argument("--distance", type=float, default=3.8)
    bjj.set_defaults(func=cmd_branchjudge)

    vs = sub.add_parser("voxscout", help="Spatial priority planner")
    vs_sub = vs.add_subparsers(dest="vs_cmd", required=True)
    vsa = vs_sub.add_parser("analyze")
    vsa.add_argument("--synthetic", action="store_true", default=True)
    vsa.set_defaults(func=cmd_voxscout)

    tr = sub.add_parser("trace", help="TraceWire provenance walk")
    tr.add_argument("kind", choices=["neuron", "synapse", "tile", "segment", "edge", "volume"])
    tr.add_argument("entity_id")
    tr.add_argument("--direction", choices=["back", "forward", "both"], default="back")
    tr.set_defaults(func=cmd_trace)

    ss = sub.add_parser("seamsmith", help="Boundary reconciliation")
    ss_sub = ss.add_subparsers(dest="ss_cmd", required=True)
    ssa = ss_sub.add_parser("analyze")
    ssa.add_argument("--synthetic", action="store_true", default=True)
    ssa.set_defaults(func=cmd_seamsmith)

    mg = sub.add_parser("morphguard", help="Morphology auditor")
    mg_sub = mg.add_subparsers(dest="mg_cmd", required=True)
    mga = mg_sub.add_parser("analyze")
    mga.add_argument("--synthetic", action="store_true", default=True)
    mga.set_defaults(func=cmd_morphguard)

    doc = sub.add_parser("doctor", help="RunDoctor pipeline diagnosis")
    doc.set_defaults(func=cmd_doctor)

    # --- Broadened commands ---
    pipe = sub.add_parser("pipeline", help="Cross-tool pipelines")
    pipe_sub = pipe.add_subparsers(dest="pipe_cmd", required=True)
    pp = pipe_sub.add_parser("plan", help="VoxScout -> AxonForge hints")
    pp.add_argument("--shape", default="64,64,64")
    pp.add_argument("--tile", default="32,32,32")
    pp.add_argument("--region", default=None, help="RegionBench region id")
    pp.set_defaults(func=cmd_pipeline_plan)
    pc = pipe_sub.add_parser("close-gaps", help="GapHound scan -> repair-plan -> TileMedic")
    pc.add_argument("--shape", default="100,100,100")
    pc.add_argument("--no-tilemedic", action="store_true")
    pc.set_defaults(func=cmd_pipeline_close)
    pr = pipe_sub.add_parser("reconcile", help="SeamSmith -> BranchJudge")
    pr.set_defaults(func=cmd_pipeline_reconcile)
    pa = pipe_sub.add_parser("audit", help="MorphGuard morphology audit")
    pa.set_defaults(func=cmd_pipeline_audit)
    pconn = pipe_sub.add_parser("connectivity", help="SynapseLens prioritize")
    pconn.set_defaults(func=cmd_pipeline_connectivity)
    pd = pipe_sub.add_parser("diff", help="DeltaGraph structural diff")
    pd.set_defaults(func=cmd_pipeline_diff)
    pm = pipe_sub.add_parser("mass-status", help="HyperDrain fleet status")
    pm.set_defaults(func=cmd_pipeline_mass)

    ep = sub.add_parser("edgeprobe", help="TraceWire edge diagnostic bundle")
    ep.add_argument("pre")
    ep.add_argument("post")
    ep.set_defaults(func=cmd_edgeprobe)

    rb = sub.add_parser("regionbench", help="Frozen synthetic benchmark regions")
    rb_sub = rb.add_subparsers(dest="rb_cmd", required=True)
    rbf = rb_sub.add_parser("freeze")
    rbf.add_argument("--force", action="store_true")
    rbf.set_defaults(func=cmd_rb_freeze)
    rbl = rb_sub.add_parser("list")
    rbl.set_defaults(func=cmd_rb_list)
    rbs = rb_sub.add_parser("smoke")
    rbs.set_defaults(func=cmd_rb_smoke)

    av = sub.add_parser("artifactvet", help="Validate npy/json before handoff")
    av_sub = av.add_subparsers(dest="av_cmd", required=True)
    avj = av_sub.add_parser("json")
    avj.add_argument("path")
    avj.add_argument("--require", nargs="*", default=[])
    avj.set_defaults(func=cmd_av_json)
    avn = av_sub.add_parser("npy")
    avn.add_argument("path")
    avn.add_argument("--shape", default=None)
    avn.add_argument("--dtype", default=None)
    avn.set_defaults(func=cmd_av_npy)


    sl = sub.add_parser("synapselens", help="Synapse evidence engine")
    sl_sub = sl.add_subparsers(dest="sl_cmd", required=True)
    sld = sl_sub.add_parser("demo")
    sld.set_defaults(func=cmd_synapselens_demo)

    dg = sub.add_parser("deltagraph", help="Connectome structural diff")
    dg_sub = dg.add_subparsers(dest="dg_cmd", required=True)
    dgd = dg_sub.add_parser("demo")
    dgd.set_defaults(func=cmd_deltagraph_demo)

    hd = sub.add_parser("hyperdrain", help="Affinity mass-production engine")
    hd_sub = hd.add_subparsers(dest="hd_cmd", required=True)
    hds = hd_sub.add_parser("status")
    hds.set_defaults(func=cmd_hyperdrain_status)

    nc = sub.add_parser("neurocache", help="Semantic cache / AxonForge tile identity")
    nc_sub = nc.add_subparsers(dest="nc_cmd", required=True)
    ncd = nc_sub.add_parser("axonforge-demo")
    ncd.add_argument("--tile", default="T-Z000-Y000-X000")
    ncd.set_defaults(func=cmd_nc_demo)
    ncs = nc_sub.add_parser("stats")
    ncs.set_defaults(func=cmd_nc_stats)

    args = ap.parse_args(argv)
    return int(args.func(args))


def cmd_tools(_a) -> int:
    import registry
    print(json.dumps(registry.as_dict_list(), indent=2))
    return 0


def cmd_switch(a) -> int:
    from axonforge.switch import SWITCH
    if a.action == "on":
        print(json.dumps(SWITCH.set_power(True), indent=2))
    elif a.action == "off":
        print(json.dumps(SWITCH.set_power(False), indent=2))
    elif a.action == "toggle":
        print(json.dumps(SWITCH.toggle(), indent=2))
    elif a.action == "autonomous":
        print(json.dumps(SWITCH.set_autonomous(True), indent=2))
    elif a.action == "manual":
        print(json.dumps(SWITCH.set_autonomous(False), indent=2))
    else:
        print(json.dumps(SWITCH.as_dict(), indent=2))
    return 0


def cmd_gaphound_scan(a) -> int:
    from gaphound.scan import GapHound
    shape = tuple(int(x) for x in a.shape.split(","))
    print(GapHound().scan(shape).format_summary())
    return 0


def cmd_gaphound_repair(_a) -> int:
    from gaphound.scan import GapHound
    print(json.dumps(GapHound().repair_plan(), indent=2))
    return 0


def cmd_tilemedic(a) -> int:
    from tilemedic.cli import cmd_recover
    return cmd_recover(a)


def cmd_branchjudge(a) -> int:
    from branchjudge.cli import cmd_judge
    return cmd_judge(a)


def cmd_voxscout(a) -> int:
    from voxscout.cli import main as vs_main
    return int(vs_main(["analyze", "--synthetic"]))


def cmd_doctor(_a) -> int:
    from runddoctor.doctor import RunDoctor
    print(RunDoctor().run().format_summary())
    return 0


def cmd_trace(a) -> int:
    from tracewire.wire import TraceWire
    tw = TraceWire()
    eid = a.entity_id if ":" in a.entity_id else f"{a.kind}:{a.entity_id}"
    candidates = [eid, a.entity_id, f"{a.kind}:{a.entity_id}"]
    if a.direction in ("back", "both"):
        print("BACK:")
        for c in candidates:
            chain = tw.trace_back(c) if hasattr(tw, "trace_back") else []
            if chain:
                for n in chain:
                    print(" ", n)
                break
        else:
            print("  (no nodes — try: python tools/tracewire.py register-demo)")
    if a.direction in ("forward", "both"):
        print("FORWARD:")
        for c in candidates:
            chain = tw.trace_forward(c) if hasattr(tw, "trace_forward") else []
            if chain:
                for n in chain:
                    print(" ", n)
                break
    return 0


def cmd_seamsmith(_a) -> int:
    from seamsmith.cli import main as ss_main
    return int(ss_main(["analyze", "--synthetic"]))


def cmd_morphguard(_a) -> int:
    from morphguard.cli import main as mg_main
    return int(mg_main(["analyze", "--synthetic"]))


def cmd_pipeline_plan(a) -> int:
    from pipeline.flows import plan_pipeline
    shape = tuple(int(x) for x in a.shape.split(","))
    tile = tuple(int(x) for x in a.tile.split(","))
    print(json.dumps(plan_pipeline(shape_zyx=shape, tile_zyx=tile, region_id=a.region), indent=2))
    return 0


def cmd_pipeline_close(a) -> int:
    from pipeline.flows import close_gaps_pipeline
    shape = tuple(int(x) for x in a.shape.split(","))
    print(json.dumps(close_gaps_pipeline(shape_zyx=shape, run_tilemedic=not a.no_tilemedic), indent=2))
    return 0


def cmd_edgeprobe(a) -> int:
    from edgeprobe.probe import edgeprobe
    print(json.dumps(edgeprobe(a.pre, a.post), indent=2))
    return 0


def cmd_rb_freeze(a) -> int:
    from regionbench.bench import ensure_frozen
    print(json.dumps([s.as_dict() for s in ensure_frozen(force=a.force)], indent=2))
    return 0


def cmd_rb_list(_a) -> int:
    from regionbench.bench import list_regions
    print(json.dumps([s.as_dict() for s in list_regions()], indent=2))
    return 0


def cmd_rb_smoke(_a) -> int:
    from regionbench.cli import cmd_smoke
    return cmd_smoke(_a)


def cmd_av_json(a) -> int:
    from _core.artifactvet import ArtifactVet
    r = ArtifactVet().vet_json(a.path, required_keys=a.require)
    print(json.dumps(r.as_dict(), indent=2))
    return 0 if r.ok else 1


def cmd_av_npy(a) -> int:
    from _core.artifactvet import ArtifactVet
    shape = tuple(int(x) for x in a.shape.split(",")) if a.shape else None
    r = ArtifactVet().vet_npy(a.path, expected_shape=shape, expected_dtype=a.dtype)
    print(json.dumps(r.as_dict(), indent=2))
    return 0 if r.ok else 1


def cmd_nc_demo(a) -> int:
    from neurocache.axonforge_id import demo_roundtrip
    print(json.dumps(demo_roundtrip(a.tile), indent=2))
    return 0


def cmd_nc_stats(_a) -> int:
    from neurocache.cache import NeuroCache
    print(json.dumps(NeuroCache().stats(), indent=2))
    return 0



def cmd_status(_a) -> int:
    try:
        from mvconnectome.toolchain import connection_status
    except ImportError:
        sys.path.insert(0, str(REPO / "src"))
        from mvconnectome.toolchain import connection_status
    print(json.dumps(connection_status(), indent=2, sort_keys=True))
    return 0 if connection_status()["ok"] else 1


def cmd_pipeline_reconcile(_a) -> int:
    from pipeline.flows import reconcile_pipeline
    print(json.dumps(reconcile_pipeline(), indent=2, default=str))
    return 0


def cmd_pipeline_audit(_a) -> int:
    from pipeline.flows import audit_pipeline
    print(json.dumps(audit_pipeline(), indent=2, default=str))
    return 0


def cmd_pipeline_connectivity(_a) -> int:
    from pipeline.flows import connectivity_pipeline
    print(json.dumps(connectivity_pipeline(), indent=2, default=str))
    return 0


def cmd_pipeline_diff(_a) -> int:
    from pipeline.flows import diff_pipeline
    print(json.dumps(diff_pipeline(), indent=2, default=str))
    return 0


def cmd_pipeline_mass(_a) -> int:
    from pipeline.flows import mass_status_pipeline
    print(json.dumps(mass_status_pipeline(), indent=2, default=str))
    return 0


def cmd_synapselens_demo(_a) -> int:
    from pipeline.flows import connectivity_pipeline
    print(json.dumps(connectivity_pipeline(), indent=2, default=str))
    return 0


def cmd_deltagraph_demo(_a) -> int:
    from pipeline.flows import diff_pipeline
    print(json.dumps(diff_pipeline(), indent=2, default=str))
    return 0


def cmd_hyperdrain_status(_a) -> int:
    from pipeline.flows import mass_status_pipeline
    print(json.dumps(mass_status_pipeline(), indent=2, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
