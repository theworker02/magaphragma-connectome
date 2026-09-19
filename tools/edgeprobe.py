#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="edgeprobe")
    ap.add_argument("pre")
    ap.add_argument("post")
    args = ap.parse_args(argv)
    from edgeprobe.probe import edgeprobe
    print(json.dumps(edgeprobe(args.pre, args.post), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
