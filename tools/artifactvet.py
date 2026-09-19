#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="artifactvet")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("json")
    j.add_argument("path")
    j.add_argument("--require", nargs="*", default=[])
    n = sub.add_parser("npy")
    n.add_argument("path")
    n.add_argument("--shape", default=None)
    n.add_argument("--dtype", default=None)
    args = ap.parse_args(argv)
    from _core.artifactvet import ArtifactVet
    vet = ArtifactVet()
    if args.cmd == "json":
        r = vet.vet_json(args.path, required_keys=args.require)
    else:
        shape = tuple(int(x) for x in args.shape.split(",")) if args.shape else None
        r = vet.vet_npy(args.path, expected_shape=shape, expected_dtype=args.dtype)
    print(json.dumps(r.as_dict(), indent=2))
    return 0 if r.ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
