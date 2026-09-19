"""SeamSmith CLI ? boundary reconciliation; never auto-merges."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parents[1]
for p in (REPO, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="seamsmith", description="SeamSmith boundary reconciliation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("analyze", "run"):
        p = sub.add_parser(name, help="Reconcile seam between two label chunks")
        p.add_argument("--labels-a", type=Path, default=None)
        p.add_argument("--labels-b", type=Path, default=None)
        p.add_argument("--axis", choices=["z", "y", "x"], default="x")
        p.add_argument("--halo", type=int, default=4)
        p.add_argument("--synthetic", action="store_true")
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(func=_analyze)
    args = ap.parse_args(argv)
    return int(args.func(args))


def _synthetic_pair(seed: int = 0, halo: int = 4):
    rng = np.random.default_rng(seed)
    za = np.zeros((16, 24, 28), dtype=np.int32)
    zb = np.zeros((16, 24, 28), dtype=np.int32)
    za[4:12, 6:18, 20:28] = 1
    zb[4:12, 6:18, 0:8] = 1
    za[2:6, 2:8, 24:28] = 2
    zb[10:14, 16:22, 0:4] = 3
    za[8:14, 10:16, 22:28] = 4
    zb[8:14, 10:16, 0:6] = 5
    _ = (rng, halo)
    return za, zb


def _analyze(args: argparse.Namespace) -> int:
    from seamsmith._core_compat import load_core
    from seamsmith.matcher import match_seam
    from seamsmith.seams import build_axis_seam
    from seamsmith.__version__ import __version__

    _, _, write_tool_receipt, ArtifactStore, ProvenanceGraph = load_core()

    if args.labels_a is not None and args.labels_b is not None:
        la = np.load(args.labels_a)
        lb = np.load(args.labels_b)
    else:
        la, lb = _synthetic_pair(args.seed, args.halo)

    report = match_seam(build_axis_seam(la, lb, axis=args.axis, halo=args.halo))
    receipts_root = REPO / "receipts" / "seamsmith"
    receipts_root.mkdir(parents=True, exist_ok=True)
    report_path = receipts_root / "seam_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + chr(10), encoding="utf-8")

    store = ArtifactStore(REPO)
    ref = store.put_json(report.to_dict(), kind="seam_report", name="seam_report.json")

    actions: dict[str, int] = {}
    for c in report.candidates:
        actions[c["recommended_action"]] = actions.get(c["recommended_action"], 0) + 1

    # Cross-tool: BranchJudge on REVIEW candidates (evidence assist, never auto-merge)
    branch_reviews = []
    try:
        from branchjudge.judge import BranchJudge
        bj = BranchJudge()
        for c in report.candidates:
            if c.get("recommended_action") != "REVIEW":
                continue
            # Build synthetic endpoint clouds from face centroids / evidence
            ev = c.get("evidence") or {}
            dist = float(ev.get("spatial_distance", 3.0))
            a_pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]], dtype=float)
            b_pts = np.array([[0.0, 0.0, 5.0 + dist], [0.2, 0.0, 10.0 + dist]], dtype=float)
            conf = float(ev.get("confidence", 0.45))
            verdict = bj.judge(
                a_pts, b_pts,
                diameter_a=2.0, diameter_b=2.0,
                seg_confidence=conf,
                at_boundary=True,
                image_agreement=float(ev.get("morphology_agreement", 0.5)),
                model_evidence=float(ev.get("direction_agreement", 0.5)),
            )
            entry = {
                "a_label": (c.get("a") or {}).get("label_id"),
                "b_label": (c.get("b") or {}).get("label_id"),
                "seam_action": c.get("recommended_action"),
                "branchjudge": verdict.as_dict(),
            }
            branch_reviews.append(entry)
            c["branchjudge"] = {
                "decision": verdict.decision,
                "merge_evidence": verdict.merge_evidence,
                "split_evidence": verdict.split_evidence,
            }
        if branch_reviews:
            br_path = receipts_root / "branchjudge_reviews.json"
            br_path.write_text(json.dumps(branch_reviews, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    except Exception as exc:
        branch_reviews = [{"error": str(exc)}]

    prov = ProvenanceGraph()
    prov.add_node("labels_a", kind="volume", attrs={"shape": list(la.shape)})
    prov.add_node("labels_b", kind="volume", attrs={"shape": list(lb.shape)})
    prov.add_node("seam_report", kind="artifact", attrs={"path": str(report_path)})
    if hasattr(prov, "record_edge"):
        prov.record_edge("labels_a", "seam_report", "seam_input")
        prov.record_edge("labels_b", "seam_report", "seam_input")

    summary = {
        "tool": "seamsmith",
        "version": __version__,
        "axis": args.axis,
        "halo": args.halo,
        "n_candidates": len(report.candidates),
        "action_counts": actions,
        "auto_merge_applied": False,
        "branchjudge_reviews": len(branch_reviews) if isinstance(branch_reviews, list) else 0,
        "report": str(report_path),
        "artifact_sha256": getattr(ref, "sha256", None),
        "provenance": prov.to_dict() if hasattr(prov, "to_dict") else prov.as_dict(),
    }
    receipt = write_tool_receipt("seamsmith", summary)
    summary["receipt"] = str(receipt)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
