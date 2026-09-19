"""Directed provenance graph with BFS ancestry and descendant walks."""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Set, Tuple


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class ProvenanceNode:
    """A node in the provenance DAG (artifact, run, decision, etc.)."""

    node_id: str
    kind: str
    label: str = ""
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "attrs": dict(self.attrs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProvenanceNode":
        return cls(
            node_id=str(data["node_id"]),
            kind=str(data.get("kind", "node")),
            label=str(data.get("label", "")),
            attrs=dict(data.get("attrs") or {}),
        )


@dataclass(frozen=True)
class ProvenanceEdge:
    parent: str
    child: str
    relation: str
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent": self.parent,
            "child": self.child,
            "relation": self.relation,
            "attrs": dict(self.attrs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProvenanceEdge":
        return cls(
            parent=str(data["parent"]),
            child=str(data["child"]),
            relation=str(data.get("relation", "derived_from")),
            attrs=dict(data.get("attrs") or {}),
        )


class ProvenanceGraph:
    """Directed provenance graph: edges point parent -> child (derivation)."""

    def __init__(self, graph_id: Optional[str] = None) -> None:
        self.graph_id = graph_id or uuid.uuid4().hex
        self.nodes: Dict[str, ProvenanceNode] = {}
        self.edges: List[ProvenanceEdge] = []
        self._out: Dict[str, List[int]] = {}
        self._in: Dict[str, List[int]] = {}

    def add_node(
        self,
        node_id: Optional[str] = None,
        *,
        kind: str = "node",
        label: str = "",
        attrs: Optional[Mapping[str, Any]] = None,
    ) -> ProvenanceNode:
        nid = node_id or uuid.uuid4().hex
        node = ProvenanceNode(node_id=nid, kind=kind, label=label, attrs=dict(attrs or {}))
        self.nodes[nid] = node
        self._out.setdefault(nid, [])
        self._in.setdefault(nid, [])
        return node

    def ensure_node(self, node_id: str, **kwargs: Any) -> ProvenanceNode:
        if node_id in self.nodes:
            return self.nodes[node_id]
        return self.add_node(node_id, **kwargs)

    def record_edge(
        self,
        parent: str,
        child: str,
        relation: str = "derived_from",
        *,
        attrs: Optional[Mapping[str, Any]] = None,
    ) -> ProvenanceEdge:
        """Record directed edge parent -> child with a relation label."""
        self.ensure_node(parent)
        self.ensure_node(child)
        edge = ProvenanceEdge(
            parent=parent, child=child, relation=relation, attrs=dict(attrs or {})
        )
        idx = len(self.edges)
        self.edges.append(edge)
        self._out.setdefault(parent, []).append(idx)
        self._in.setdefault(child, []).append(idx)
        return edge

    def parents_of(self, node_id: str) -> List[Tuple[str, str]]:
        return [(self.edges[i].parent, self.edges[i].relation) for i in self._in.get(node_id, [])]

    def children_of(self, node_id: str) -> List[Tuple[str, str]]:
        return [(self.edges[i].child, self.edges[i].relation) for i in self._out.get(node_id, [])]

    def walk_back(self, start: str, *, max_depth: Optional[int] = None) -> List[str]:
        """BFS toward parents (ancestors), excluding start."""
        return self._bfs(start, reverse=True, max_depth=max_depth)

    def walk_forward(self, start: str, *, max_depth: Optional[int] = None) -> List[str]:
        """BFS toward children (descendants), excluding start."""
        return self._bfs(start, reverse=False, max_depth=max_depth)

    def _bfs(self, start: str, *, reverse: bool, max_depth: Optional[int]) -> List[str]:
        if start not in self.nodes and start not in self._out and start not in self._in:
            return []
        seen: Set[str] = {start}
        order: List[str] = []
        q: Deque[Tuple[str, int]] = deque([(start, 0)])
        while q:
            cur, depth = q.popleft()
            if max_depth is not None and depth >= max_depth:
                continue
            if reverse:
                nbrs = [self.edges[i].parent for i in self._in.get(cur, [])]
            else:
                nbrs = [self.edges[i].child for i in self._out.get(cur, [])]
            for nxt in nbrs:
                if nxt in seen:
                    continue
                seen.add(nxt)
                order.append(nxt)
                q.append((nxt, depth + 1))
        return order

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProvenanceGraph":
        g = cls(graph_id=str(data.get("graph_id") or uuid.uuid4().hex))
        for nd in data.get("nodes") or []:
            node = ProvenanceNode.from_dict(nd)
            g.nodes[node.node_id] = node
            g._out.setdefault(node.node_id, [])
            g._in.setdefault(node.node_id, [])
        for ed in data.get("edges") or []:
            edge = ProvenanceEdge.from_dict(ed)
            idx = len(g.edges)
            g.edges.append(edge)
            g._out.setdefault(edge.parent, []).append(idx)
            g._in.setdefault(edge.child, []).append(idx)
            g._out.setdefault(edge.child, [])
            g._in.setdefault(edge.parent, [])
        return g

    def save(self, path: Optional[Path] = None) -> Path:
        out = Path(path) if path is not None else (
            _project_root() / "receipts" / "provenance" / f"{self.graph_id}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + chr(10)
        out.write_text(body, encoding="utf-8")
        (out.parent / "latest.json").write_text(body, encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Path) -> "ProvenanceGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
