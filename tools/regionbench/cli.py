"""RegionBench CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="regionbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze", help="Materialize frozen synthetic regions")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_freeze)
    ls = sub.add_parser("list", help="List frozen regions")
    ls.set_defaults(func=cmd_list)
    sm = sub.add_parser("smoke", help="Run VoxScout metrics across all regions")
    sm.set_defaults(func=cmd_smoke)
    args = ap.parse_args(argv)
    return int(args.func(args))


def cmd_freeze(a) -> int:
    from regionbench.bench import ensure_frozen
    specs = ensure_frozen(force=a.force)
    print(json.dumps([s.as_dict() for s in specs], indent=2))
    return 0


def cmd_list(_a) -> int:
    from regionbench.bench import list_regions
    print(json.dumps([s.as_dict() for s in list_regions()], indent=2))
    return 0


def cmd_smoke(_a) -> int:
    from regionbench.bench import RegionBench
    from voxscout.scout import VoxScout

    def metrics(rid, vol):
        tile = (16, 16, 16) if min(vol.shape) >= 32 else tuple(max(8, s // 2) for s in vol.shape)
        pm = VoxScout(tile).analyze(vol)
        return {"n_tiles": len(pm.tiles), "class_counts": pm.class_counts()}

    results = RegionBench().run("voxscout", metrics)
    print(json.dumps([r.as_dict() for r in results], indent=2))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
