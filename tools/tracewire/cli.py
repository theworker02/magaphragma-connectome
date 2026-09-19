"""TraceWire CLI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.tracewire.wire import PIPELINE_STAGES, TraceWire
from tools.tracewire.__version__ import __version__


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tracewire", description="Scientific provenance debugger")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--graph", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trace", help="Trace provenance for an entity")
    t.add_argument("kind", choices=["neuron", "synapse", "tile", "segment", "volume", "edge", "all"])
    t.add_argument("entity_id")
    t.add_argument("--direction", choices=["back", "forward", "both"], default="back")

    r = sub.add_parser("register-demo", help="Register a demo pipeline chain and persist")
    r.add_argument("--prefix", default="demo")

    args = ap.parse_args(argv)
    tw = TraceWire(path=args.graph) if args.graph else TraceWire()

    if args.cmd == "register-demo":
        ids = {stage: f"{args.prefix}-{stage}-001" for stage in PIPELINE_STAGES}
        chain = tw.register_chain(ids)
        path = tw.persist()
        print(tw.format_ascii([tw._node_dict(n) for n in chain]))
        print(f"persisted: {path}")
        return 0

    if args.cmd == "trace":
        eid = args.entity_id
        if args.direction in {"back", "both"}:
            nodes = tw.trace_back(eid)
            print("=== TRACE BACK ===")
            print(tw.format_ascii(nodes) if nodes else f"(no node: {eid})")
        if args.direction in {"forward", "both"}:
            nodes = tw.trace_forward(eid)
            print("=== TRACE FORWARD ===")
            print(tw.format_ascii(nodes) if nodes else f"(no node: {eid})")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
