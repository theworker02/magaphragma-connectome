"""RunDoctor CLI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(TOOLS)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="runddoctor")
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "doctor"])
    args = ap.parse_args(argv)
    from runddoctor.doctor import RunDoctor
    print(RunDoctor().run().format_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
