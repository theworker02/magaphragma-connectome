"""NeuroCache CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(TOOLS)]

from neurocache.cache import NeuroCache
from neurocache.identity import ScientificIdentity
from neurocache.__version__ import __version__


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="neurocache", description="Semantic computation cache")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("--root", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    lu = sub.add_parser("lookup", help="Lookup by identity JSON")
    lu.add_argument("identity", type=Path)

    pu = sub.add_parser("put", help="Put payload for identity")
    pu.add_argument("identity", type=Path)
    pu.add_argument("payload", type=Path)

    st = sub.add_parser("stats", help="Cache statistics")

    af = sub.add_parser("axonforge-demo", help="Demo put/lookup for a tile inference identity")
    af.add_argument("--tile", default="T-Z000-Y000-X000")

    afp = sub.add_parser("axonforge-put", help="Put tile inference result")
    afp.add_argument("identity", type=Path)
    afp.add_argument("payload", type=Path)

    afl = sub.add_parser("axonforge-lookup", help="Lookup tile inference by identity JSON")
    afl.add_argument("identity", type=Path)

    args = ap.parse_args(argv)
    cache = NeuroCache(root=args.root) if args.root else NeuroCache()

    if args.cmd == "lookup":
        ident = ScientificIdentity.from_dict(json.loads(args.identity.read_text(encoding="utf-8")))
        hit = cache.lookup(ident)
        print(json.dumps({"hit": hit is not None, "artifact": hit}, indent=2, sort_keys=True))
        return 0 if hit is not None else 2

    if args.cmd == "put":
        ident = ScientificIdentity.from_dict(json.loads(args.identity.read_text(encoding="utf-8")))
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        man = cache.put(ident, payload)
        print(json.dumps(man, indent=2, sort_keys=True))
        return 0

    if args.cmd == "stats":
        print(json.dumps(cache.stats(), indent=2, sort_keys=True))
        return 0

    if args.cmd == "axonforge-demo":
        from neurocache.axonforge_id import demo_roundtrip
        print(json.dumps(demo_roundtrip(args.tile), indent=2, sort_keys=True))
        return 0

    if args.cmd == "axonforge-put":
        from neurocache.axonforge_id import put_tile_result
        ident = ScientificIdentity.from_dict(json.loads(args.identity.read_text(encoding="utf-8")))
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        print(json.dumps(put_tile_result(ident, payload, cache=cache), indent=2, sort_keys=True))
        return 0

    if args.cmd == "axonforge-lookup":
        from neurocache.axonforge_id import lookup_tile_result
        ident = ScientificIdentity.from_dict(json.loads(args.identity.read_text(encoding="utf-8")))
        hit = lookup_tile_result(ident, cache=cache)
        print(json.dumps({"hit": hit is not None, "payload": hit}, indent=2, sort_keys=True))
        return 0 if hit is not None else 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
