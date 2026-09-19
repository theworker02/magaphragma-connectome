#!/usr/bin/env python3
"""Pipeline CLI shim."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--shape", default="64,64,64")
    p.add_argument("--tile", default="32,32,32")
    p.add_argument("--region", default=None)
    p.set_defaults(func=cmd_plan)
    c = sub.add_parser("close-gaps")
    c.add_argument("--shape", default="100,100,100")
    c.add_argument("--no-tilemedic", action="store_true")
    c.set_defaults(func=cmd_close)
    args = ap.parse_args(argv)
    return int(args.func(args))

def cmd_plan(a) -> int:
    from pipeline.flows import plan_pipeline
    shape = tuple(int(x) for x in a.shape.split(","))
    tile = tuple(int(x) for x in a.tile.split(","))
    print(json.dumps(plan_pipeline(shape_zyx=shape, tile_zyx=tile, region_id=a.region), indent=2))
    return 0

def cmd_close(a) -> int:
    from pipeline.flows import close_gaps_pipeline
    shape = tuple(int(x) for x in a.shape.split(","))
    print(json.dumps(close_gaps_pipeline(shape_zyx=shape, run_tilemedic=not a.no_tilemedic), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
