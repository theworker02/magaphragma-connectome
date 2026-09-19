"""SynapseLens CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.synapselens.evidence import CandidateSynapse, build_report
from tools.synapselens.prioritize import prioritize
from tools.synapselens.__version__ import __version__

try:
    from tools._core import write_tool_receipt
except Exception:  # pragma: no cover
    def write_tool_receipt(tool, payload):  # type: ignore
        root = REPO / "receipts" / "tools" / tool
        root.mkdir(parents=True, exist_ok=True)
        path = root / "latest.json"
        path.write_text(json.dumps({"tool": tool, **payload}, indent=2) + chr(10), encoding="utf-8")
        return path


def _load_candidates(path: Path) -> list[CandidateSynapse]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("candidates", [])
    out: list[CandidateSynapse] = []
    for item in items:
        xyz = item.get("xyz") or [0.0, 0.0, 0.0]
        out.append(
            CandidateSynapse(
                pre_id=str(item["pre_id"]),
                post_id=str(item["post_id"]),
                xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                prediction_confidence=float(item.get("prediction_confidence", 0.0)),
                local_image_evidence=float(item.get("local_image_evidence", 0.0)),
                segmentation_confidence=float(item.get("segmentation_confidence", 0.0)),
                boundary_status=str(item.get("boundary_status", "uncertain")),
                morphological_context=float(item.get("morphological_context", 0.5)),
                model_agreement=float(item.get("model_agreement", 0.5)),
                provenance=dict(item.get("provenance") or {}),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="synapselens", description="Synapse evidence engine")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prioritize", help="Classify + prioritize candidates from JSON")
    p.add_argument("input", type=Path, help="JSON file of candidate synapses")
    p.add_argument("--receipt", action="store_true", help="Write tool receipt")

    args = ap.parse_args(argv)
    if args.cmd == "prioritize":
        cands = _load_candidates(args.input)
        ranked = prioritize(cands)
        report = build_report(cands, ranked=ranked)
        out = report.to_dict()
        print(json.dumps(out, indent=2, sort_keys=True))
        if args.receipt:
            write_tool_receipt("synapselens", out)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
