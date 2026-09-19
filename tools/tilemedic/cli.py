"""TileMedic CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(TOOLS)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tilemedic")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recover")
    r.add_argument("--failed", action="store_true", help="Recover from AxonForge rejected tiles")
    r.add_argument("--error", default="")
    r.add_argument("--tile", default="T-Z000-Y000-X000")
    r.add_argument("--code", default="")
    r.add_argument("--batch-size", type=int, default=8)
    r.set_defaults(func=cmd_recover)
    args = ap.parse_args(argv)
    return args.func(args)


def cmd_recover(args) -> int:
    from tilemedic.medic import TileMedic
    failures = []
    if args.failed:
        wg = REPO / "receipts" / "axonforge" / "work_graph.json"
        if wg.exists():
            tasks = json.loads(wg.read_text(encoding="utf-8")).get("tasks", {})
            for k, v in tasks.items():
                if v.get("status") == "REJECTED":
                    failures.append({"tile_key": k, "error": v.get("reason", "rejected"), "code": "REJECTED"})
        if not failures:
            failures.append({"tile_key": args.tile, "error": "oom", "code": "OOM"})
    else:
        failures.append({"tile_key": args.tile, "error": args.error or "oom", "code": args.code or "OOM"})
    medic = TileMedic()
    out = []
    for f in failures:
        out.append(medic.recover(f, config={"batch_size": args.batch_size, "checkpoint": "frozen"}).as_dict())
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
