"""BranchJudge CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(TOOLS)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="branchjudge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--demo", action="store_true")
    j.add_argument("--seg-confidence", type=float, default=0.45)
    j.add_argument("--distance", type=float, default=3.8)
    j.set_defaults(func=cmd_judge)
    args = ap.parse_args(argv)
    return args.func(args)


def cmd_judge(args) -> int:
    from branchjudge.judge import BranchJudge
    # Synthetic near-aligned endpoints
    a = np.array([[0, 0, 0], [0, 0, 5], [0, 0, 10]], dtype=float)
    b = np.array([[0, 0, 10 + args.distance], [0, 0, 15 + args.distance], [0.2, 0, 20 + args.distance]], dtype=float)
    v = BranchJudge().judge(a, b, diameter_a=2.0, diameter_b=2.1, seg_confidence=args.seg_confidence,
                            at_boundary=True, image_agreement=0.7, model_evidence=0.65)
    print(v.format_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
