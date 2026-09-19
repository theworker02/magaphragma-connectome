"""Walk TraceWire for an edge and dump a small diagnostic bundle."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "receipts" / "edgeprobe"


def edgeprobe(pre: str, post: str, *, ensure_demo: bool = True) -> dict[str, Any]:
    from tracewire.wire import TraceWire, PIPELINE_STAGES

    tw = TraceWire()
    graph_path = REPO / "receipts" / "tracewire" / "graph.json"
    if ensure_demo and (not graph_path.exists() or not tw.graph.nodes if hasattr(tw.graph, "nodes") else True):
        # Register a minimal edge-linked demo chain if empty
        try:
            n = len(getattr(tw.graph, "nodes", {}) or {})
        except Exception:
            n = 0
        if n == 0:
            prefix = f"ep-{pre}-{post}"
            ids = {
                "volume": f"{prefix}-volume-001",
                "tile": f"{prefix}-tile-001",
                "inference": f"{prefix}-infer-001",
                "segment": f"{prefix}-seg-001",
                "synapse": f"{prefix}-syn-001",
                "graph_edge": f"edge:{pre}->{post}",
                "neuron": f"neuron:{pre}",
            }
            # fill required stages with placeholders
            for stage in PIPELINE_STAGES:
                ids.setdefault(stage, f"{prefix}-{stage}-001")
            tw.register_chain(ids, attrs={
                "graph_edge": {"pre": pre, "post": post},
                "neuron": {"label": pre},
                "synapse": {"pre": pre, "post": post},
            })
            tw.persist()

    edge_id = f"edge:{pre}->{post}"
    # also try graph_edge style ids present in graph
    candidates = [edge_id, f"graph_edge:{pre}->{post}", post, pre]
    back = forward = []
    used = None
    for c in candidates:
        back = tw.trace_back(c)
        forward = tw.trace_forward(c)
        if len(back) + len(forward) > 1:
            used = c
            break
    if used is None:
        prefix = f"ep-{pre}-{post}"
        ids = {stage: f"{prefix}-{stage}-001" for stage in PIPELINE_STAGES}
        ids["graph_edge"] = edge_id
        ids["neuron"] = f"neuron:{pre}"
        ids["synapse"] = f"syn:{pre}->{post}"
        tw.register_chain(ids, attrs={
            "graph_edge": {"pre": pre, "post": post},
            "synapse": {"pre": pre, "post": post},
            "neuron": {"id": pre},
        })
        # Link post neuron as sibling leaf
        tw._add_node(f"neuron:{post}", "neuron", {"id": post})
        try:
            tw._link(edge_id, f"neuron:{post}")
        except Exception:
            pass
        tw.persist()
        used = edge_id
        back = tw.trace_back(used)
        forward = tw.trace_forward(used)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    bundle_dir = OUT / f"{pre}__{post}__{stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema": "edgeprobe/v1",
        "pre": pre,
        "post": post,
        "entity_id": used,
        "n_back": len(back),
        "n_forward": len(forward),
        "back": back,
        "forward": forward,
        "ascii_back": tw.format_ascii(back),
        "ascii_forward": tw.format_ascii(forward),
    }
    (bundle_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    (bundle_dir / "back.txt").write_text(summary["ascii_back"] + "\\n", encoding="utf-8")
    (bundle_dir / "forward.txt").write_text(summary["ascii_forward"] + "\\n", encoding="utf-8")
    # copy graph snapshot pointer
    if graph_path.exists():
        (bundle_dir / "graph_ref.json").write_text(
            json.dumps({"graph": str(graph_path), "sha_hint": graph_path.stat().st_size}, indent=2) + "\\n",
            encoding="utf-8",
        )
    latest = OUT / "latest.json"
    latest.write_text(json.dumps({**summary, "bundle": str(bundle_dir)}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    summary["bundle"] = str(bundle_dir)
    return summary
