"""DeltaGraph CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.deltagraph.causes import infer
from tools.deltagraph.diff import Build, DeltaGraph
from tools.deltagraph.__version__ import __version__

try:
    from tools._core import write_tool_receipt
except Exception:  # pragma: no cover
    def write_tool_receipt(tool, payload):  # type: ignore
        root = REPO / "receipts" / "deltagraph"
        root.mkdir(parents=True, exist_ok=True)
        path = root / "latest.json"
        path.write_text(json.dumps({"tool": tool, **payload}, indent=2) + chr(10), encoding="utf-8")
        return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="deltagraph", description="Connectome structural diff")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diff", help="Diff two build JSON files")
    d.add_argument("build_a", type=Path)
    d.add_argument("build_b", type=Path)
    d.add_argument("--infer-causes", action="store_true")
    d.add_argument("--receipt", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "diff":
        a = Build.from_dict(json.loads(args.build_a.read_text(encoding="utf-8")))
        b = Build.from_dict(json.loads(args.build_b.read_text(encoding="utf-8")))
        delta = DeltaGraph.diff(a, b)
        out: dict = {"delta": delta.to_dict()}
        if args.infer_causes:
            out["causes"] = infer(delta, a.meta, b.meta)
        print(json.dumps(out, indent=2, sort_keys=True))
        if args.receipt:
            # Dedicated receipts/deltagraph/ plus tools receipt
            dest = REPO / "receipts" / "deltagraph"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "latest.json").write_text(
                json.dumps(out, indent=2, sort_keys=True) + chr(10),
                encoding="utf-8",
                newline="\n",
            )
            write_tool_receipt("deltagraph", out)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
