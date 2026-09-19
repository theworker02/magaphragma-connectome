"""MorphGuard CLI ? morphology auditor (heuristic review flags only)."""
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
    ap = argparse.ArgumentParser(prog="morphguard", description="MorphGuard morphology auditor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("analyze", "run"):
        p = sub.add_parser(name, help="Audit labels / skeletons")
        p.add_argument("--labels", type=Path, default=None)
        p.add_argument("--synthetic", action="store_true")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--tiny-threshold", type=int, default=20)
        p.set_defaults(func=_analyze)
    args = ap.parse_args(argv)
    return int(args.func(args))


def _synthetic(seed: int = 0):
    from morphguard.audit import SkeletonGraph

    rng = np.random.default_rng(seed)
    labels = np.zeros((24, 32, 32), dtype=np.int32)
    labels[4:20, 10:14, 10:14] = 1
    labels[2:5, 20:24, 20:24] = 2
    labels[18:22, 2:6, 2:6] = 2
    labels[12, 28, 28] = 3
    labels[12, 28, 27] = 3
    labels[0:3, 0:2, 8:10] = 4
    labels[6:10, 16:22, 16:22] = 5
    labels[10:18, 18:20, 18:20] = 5
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [20, 0, 0], [3, 1, 0], [3, 2, 0], [3, -1, 0]],
        dtype=np.float64,
    )
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [3, 7], [3, 7]], dtype=np.int64)
    t_nodes = np.asarray([[i * 0.2, np.sin(i) * 2.0, np.cos(i) * 2.0] for i in range(20)], dtype=np.float64)
    t_edges = np.array([[i, i + 1] for i in range(len(t_nodes) - 1)], dtype=np.int64)
    graphs = [
        SkeletonGraph(nodes=nodes, edges=edges, label_id="sk1"),
        SkeletonGraph(nodes=t_nodes, edges=t_edges, label_id="sk_tort"),
    ]
    _ = rng
    return labels, graphs


def _analyze(args: argparse.Namespace) -> int:
    from morphguard._core_compat import load_core
    from morphguard.audit import MorphGuard
    from morphguard.__version__ import __version__

    _, _, write_tool_receipt, ArtifactStore, ProvenanceGraph = load_core()

    skeletons = None
    if args.labels is not None:
        labels = np.load(args.labels)
    else:
        labels, skeletons = _synthetic(args.seed)

    amap = MorphGuard().analyze(labels=labels, skeletons=skeletons, tiny_threshold=args.tiny_threshold)
    receipts_root = REPO / "receipts" / "morphguard"
    receipts_root.mkdir(parents=True, exist_ok=True)
    out_path = receipts_root / "anomaly_map.json"
    amap.save(out_path)

    store = ArtifactStore(REPO)
    ref = store.put_json(amap.to_dict(), kind="anomaly_map", name="anomaly_map.json")

    prov = ProvenanceGraph()
    prov.add_node("labels", kind="volume", attrs={"shape": list(labels.shape)})
    prov.add_node("anomaly_map", kind="artifact", attrs={"path": str(out_path)})
    if hasattr(prov, "record_edge"):
        prov.record_edge("labels", "anomaly_map", "audited_by_morphguard")

    summary = {
        "tool": "morphguard",
        "version": __version__,
        "n_anomalies": len(amap.anomalies),
        "kind_counts": amap.kind_counts(),
        "review_queue_size": len(amap.review_queue()),
        "disclaimer": "Heuristic morphology flags ? not definitive biology errors.",
        "anomaly_map": str(out_path),
        "artifact_sha256": getattr(ref, "sha256", None),
        "provenance": prov.to_dict() if hasattr(prov, "to_dict") else prov.as_dict(),
    }
    receipt = write_tool_receipt("morphguard", summary)
    summary["receipt"] = str(receipt)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
