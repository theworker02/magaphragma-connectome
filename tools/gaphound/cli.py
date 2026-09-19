"""GapHound CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(TOOLS)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gaphound")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--shape", default="100,100,100")
    s.add_argument("--manifest", default="")
    s.set_defaults(func=cmd_scan)
    r = sub.add_parser("repair-plan")
    r.set_defaults(func=cmd_repair)
    args = ap.parse_args(argv)
    return args.func(args)


def cmd_scan(args) -> int:
    from gaphound.scan import GapHound
    shape = tuple(int(x) for x in args.shape.split(","))
    man = Path(args.manifest) if args.manifest else None
    report = GapHound().scan(shape, manifest=man)
    print(report.format_summary())
    return 0


def cmd_repair(_args) -> int:
    from gaphound.scan import GapHound
    plan = GapHound().repair_plan()
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
