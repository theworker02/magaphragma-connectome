"""Provenance wire: register chains and walk forward/back."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from tools._core import ProvenanceGraph
except Exception:  # pragma: no cover
    ProvenanceGraph = None  # type: ignore


PIPELINE_STAGES: tuple[str, ...] = (
    "volume",
    "preprocess",
    "tile",
    "model_checkpoint",
    "inference",
    "segment",
    "synapse",
    "graph_edge",
    "neuron",
)


def _default_graph_path() -> Path:
    return Path(__file__).resolve().parents[2] / "receipts" / "tracewire" / "graph.json"


def _make_local_graph():
    from collections import defaultdict, deque
    from dataclasses import asdict, dataclass, field

    @dataclass
    class _Node:
        node_id: str
        kind: str
        meta: dict = field(default_factory=dict)

        def as_dict(self) -> dict:
            return asdict(self)

    class LocalGraph:
        def __init__(self) -> None:
            self.nodes: dict[str, _Node] = {}
            self.fwd: dict[str, list[tuple[str, str]]] = defaultdict(list)
            self.back: dict[str, list[tuple[str, str]]] = defaultdict(list)

        def add_node(self, node_id: str, kind: str = "node", **meta: Any):
            n = _Node(node_id, kind, meta)
            self.nodes[node_id] = n
            return n

        def link(self, parent: str, child: str, relation: str = "produces") -> None:
            self.fwd[parent].append((child, relation))
            self.back[child].append((parent, relation))

        def _walk(self, start: str, edges: dict[str, list[tuple[str, str]]]) -> list[str]:
            seen: set[str] = set()
            order: list[str] = []
            q = __import__("collections").deque([start])
            while q:
                cur = q.popleft()
                if cur in seen:
                    continue
                seen.add(cur)
                order.append(cur)
                for nxt, _rel in edges.get(cur, []):
                    if nxt not in seen:
                        q.append(nxt)
            return order

        def walk_forward(self, start: str) -> list[str]:
            return self._walk(start, self.fwd)

        def walk_back(self, start: str) -> list[str]:
            return self._walk(start, self.back)

        def to_dict(self) -> dict[str, Any]:
            return {
                "nodes": {k: v.as_dict() for k, v in self.nodes.items()},
                "edges": [
                    {"parent": p, "child": c, "relation": r}
                    for p, lst in self.fwd.items()
                    for c, r in lst
                ],
            }

        @classmethod
        def from_dict(cls, raw: dict[str, Any]) -> "LocalGraph":
            g = cls()
            for nid, n in (raw.get("nodes") or {}).items():
                if isinstance(n, dict):
                    kind = n.get("kind", "unknown")
                    meta = dict(n.get("meta") or n.get("attrs") or {})
                    if "stage" in n:
                        meta.setdefault("stage", n["stage"])
                    g.add_node(nid, kind, **meta)
            for e in raw.get("edges") or []:
                g.link(e["parent"], e["child"], e.get("relation", "produces"))
            return g

        def save(self, path: Path | str) -> Path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True) + chr(10),
                encoding="utf-8",
                newline="\n",
            )
            return path

        @classmethod
        def load(cls, path: Path | str) -> "LocalGraph":
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    return LocalGraph()


class TraceWire:
    """Wraps ProvenanceGraph (or local graph) for typical pipeline chains."""

    def __init__(self, graph: Any | None = None, *, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else _default_graph_path()
        if graph is not None:
            self.graph = graph
        elif self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.graph = self._load_graph(raw)
        else:
            self.graph = self._new_graph()

    def _new_graph(self) -> Any:
        if ProvenanceGraph is not None:
            try:
                return ProvenanceGraph()
            except Exception:
                pass
        return _make_local_graph()

    def _load_graph(self, raw: dict[str, Any]) -> Any:
        if ProvenanceGraph is not None and hasattr(ProvenanceGraph, "from_dict"):
            try:
                return ProvenanceGraph.from_dict(raw)
            except Exception:
                pass
        g = _make_local_graph()
        return type(g).from_dict(raw)

    def _api(self) -> str:
        """Detect graph API flavor: attrs | meta | local."""
        import inspect

        try:
            sig = inspect.signature(self.graph.add_node)
            params = sig.parameters
            if "attrs" in params:
                return "attrs"
            if any(p.kind == p.VAR_KEYWORD for p in params.values()):
                return "meta"
        except Exception:
            pass
        if hasattr(self.graph, "link") and hasattr(self.graph, "walk_back"):
            return "meta"
        return "attrs"

    def _add_node(self, eid: str, stage: str, extra: dict[str, Any] | None = None) -> None:
        extra = dict(extra or {})
        extra.setdefault("stage", stage)
        api = self._api()
        if api == "attrs":
            # tools._core.provenance.ProvenanceGraph
            if eid in getattr(self.graph, "nodes", {}):
                node = self.graph.nodes[eid]
                if hasattr(node, "attrs"):
                    node.attrs.update(extra)
                return
            self.graph.add_node(eid, kind=stage, label=stage, attrs=extra)
        else:
            # underscore / local: add_node(id, kind, **meta)
            self.graph.add_node(eid, stage, **extra) if self._positional_kind() else self.graph.add_node(
                eid, kind=stage, **extra
            )

    def _positional_kind(self) -> bool:
        import inspect

        try:
            params = list(inspect.signature(self.graph.add_node).parameters.values())
            # self, node_id, kind, **meta
            return len(params) >= 3 and params[2].kind == params[2].POSITIONAL_OR_KEYWORD
        except Exception:
            return True

    def _link(self, parent: str, child: str) -> None:
        if hasattr(self.graph, "link"):
            self.graph.link(parent, child, relation="produces")
        elif hasattr(self.graph, "record_edge"):
            self.graph.record_edge(parent, child, relation="produces")
        else:
            raise RuntimeError("graph cannot link nodes")

    def register_chain(
        self,
        ids: dict[str, str],
        *,
        attrs: dict[str, dict[str, Any]] | None = None,
    ) -> list[str]:
        """Register typical pipeline: volume->...->neuron. Returns ordered entity ids."""
        attrs = attrs or {}
        ordered: list[str] = []
        for stage in PIPELINE_STAGES:
            if stage not in ids:
                continue
            eid = str(ids[stage])
            self._add_node(eid, stage, attrs.get(stage))
            ordered.append(eid)
        for i in range(len(ordered) - 1):
            self._link(ordered[i], ordered[i + 1])
        return ordered

    def persist(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.graph, "save"):
            try:
                return self.graph.save(self.path)
            except TypeError:
                pass
        payload = self.graph.to_dict() if hasattr(self.graph, "to_dict") else {"nodes": {}, "edges": []}
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + chr(10),
            encoding="utf-8",
            newline="\n",
        )
        return self.path

    def _node_dict(self, nid: str) -> dict[str, Any]:
        node = getattr(self.graph, "nodes", {}).get(nid)
        if node is None:
            return {"id": nid, "node_id": nid, "stage": "?", "kind": "?"}
        if hasattr(node, "as_dict"):
            d = node.as_dict()
        elif hasattr(node, "to_dict"):
            d = node.to_dict()
        else:
            d = {"node_id": nid, "kind": getattr(node, "kind", "?")}
        d["id"] = d.get("node_id", nid)
        meta = d.get("meta") or d.get("attrs") or getattr(node, "meta", None) or getattr(node, "attrs", {}) or {}
        d["stage"] = meta.get("stage") or d.get("kind")
        return d

    def _walk_back_ids(self, entity_id: str) -> list[str]:
        if hasattr(self.graph, "walk_back"):
            ids = list(self.graph.walk_back(entity_id))
            # Some graphs exclude start; some include it at front.
            if not ids or ids[0] != entity_id:
                ids = [entity_id] + ids
            return list(reversed(ids))
        # attrs API parents_of
        if hasattr(self.graph, "parents_of"):
            seen: set[str] = set()
            order: list[str] = []

            def walk(nid: str) -> None:
                if nid in seen:
                    return
                seen.add(nid)
                for parent, _rel in self.graph.parents_of(nid):
                    walk(parent)
                order.append(nid)

            walk(entity_id)
            return order
        return [entity_id]

    def _walk_forward_ids(self, entity_id: str) -> list[str]:
        if hasattr(self.graph, "walk_forward"):
            ids = list(self.graph.walk_forward(entity_id))
            if not ids or ids[0] != entity_id:
                ids = [entity_id] + ids
            # Dedup preserve
            seen: set[str] = set()
            out: list[str] = []
            for nid in ids:
                if nid not in seen:
                    seen.add(nid)
                    out.append(nid)
            return out
        if hasattr(self.graph, "children_of"):
            seen: set[str] = set()
            order: list[str] = []

            def walk(nid: str) -> None:
                if nid in seen:
                    return
                seen.add(nid)
                order.append(nid)
                for child, _rel in sorted(self.graph.children_of(nid)):
                    walk(child)

            walk(entity_id)
            return order
        return [entity_id]

    def trace_back(self, entity_id: str) -> list[dict[str, Any]]:
        """Ordered ancestors including entity: root ... entity."""
        ids = self._walk_back_ids(entity_id)
        seen: set[str] = set()
        ordered: list[str] = []
        for nid in ids:
            if nid not in seen:
                seen.add(nid)
                ordered.append(nid)
        return [self._node_dict(n) for n in ordered]

    def trace_forward(self, entity_id: str) -> list[dict[str, Any]]:
        """Ordered descendants including entity: entity ... leaves."""
        ids = self._walk_forward_ids(entity_id)
        seen: set[str] = set()
        ordered: list[str] = []
        for nid in ids:
            if nid not in seen:
                seen.add(nid)
                ordered.append(nid)
        return [self._node_dict(n) for n in ordered]

    def format_ascii(self, nodes: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for i, node in enumerate(nodes):
            stage = node.get("stage") or node.get("kind") or "?"
            nid = node.get("id") or node.get("node_id") or "?"
            if i == 0:
                lines.append(f"{stage}: {nid}")
            else:
                prefix = "`-- " if i == len(nodes) - 1 else "|-- "
                lines.append(f"{prefix}{stage}: {nid}")
        return chr(10).join(lines)
