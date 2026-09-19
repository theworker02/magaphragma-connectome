"""Structural connectome diff between two builds."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Build:
    """Connectome build snapshot (BuildA / BuildB)."""

    neurons: dict[str, dict[str, Any]] = field(default_factory=dict)
    synapses: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    components: list[list[str]] = field(default_factory=list)
    degrees: dict[str, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Build":
        edges_raw = data.get("edges") or []
        edges: list[tuple[str, str]] = []
        for e in edges_raw:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                edges.append((str(e[0]), str(e[1])))
            elif isinstance(e, dict):
                edges.append((str(e["pre"]), str(e["post"])))
        degrees = {str(k): int(v) for k, v in (data.get("degrees") or {}).items()}
        if not degrees and edges:
            deg: dict[str, int] = {}
            for pre, post in edges:
                deg[pre] = deg.get(pre, 0) + 1
                deg[post] = deg.get(post, 0) + 1
            degrees = deg
        return cls(
            neurons={str(k): dict(v) for k, v in (data.get("neurons") or {}).items()},
            synapses={str(k): dict(v) for k, v in (data.get("synapses") or {}).items()},
            edges=edges,
            components=[list(map(str, c)) for c in (data.get("components") or [])],
            degrees=degrees,
            meta=dict(data.get("meta") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "neurons": self.neurons,
            "synapses": self.synapses,
            "edges": [list(e) for e in self.edges],
            "components": self.components,
            "degrees": self.degrees,
            "meta": self.meta,
        }


def _centroid(attrs: dict[str, Any]) -> tuple[float, float, float] | None:
    for key in ("centroid", "xyz", "center"):
        v = attrs.get(key)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            return (float(v[0]), float(v[1]), float(v[2]))
    if all(k in attrs for k in ("x", "y", "z")):
        return (float(attrs["x"]), float(attrs["y"]), float(attrs["z"]))
    return None


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Deterministic overlap heuristic in [0, 1] from attrs."""
    if "overlap" in a and isinstance(a["overlap"], (int, float)):
        return float(a["overlap"])
    sa = set(a.get("voxel_ids") or a.get("member_ids") or [])
    sb = set(b.get("voxel_ids") or b.get("member_ids") or [])
    if sa or sb:
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / float(len(sa | sb))
    ca, cb = _centroid(a), _centroid(b)
    if ca is None or cb is None:
        return 0.0
    scale = float(a.get("merge_radius", b.get("merge_radius", 50.0)))
    d = _dist(ca, cb)
    return max(0.0, 1.0 - d / max(scale, 1e-9))


@dataclass
class DeltaResult:
    neurons_added: list[str] = field(default_factory=list)
    neurons_removed: list[str] = field(default_factory=list)
    objects_split: list[dict[str, Any]] = field(default_factory=list)
    objects_merged: list[dict[str, Any]] = field(default_factory=list)
    synapses_added: list[str] = field(default_factory=list)
    synapses_removed: list[str] = field(default_factory=list)
    edges_changed: dict[str, list[list[str]]] = field(default_factory=dict)
    component_changes: dict[str, Any] = field(default_factory=dict)
    degree_changes: dict[str, dict[str, int]] = field(default_factory=dict)
    spatial_changes: list[dict[str, Any]] = field(default_factory=list)
    confidence_changes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeltaGraph:
    """Diff Build A vs Build B with centroid+overlap split/merge heuristics."""

    @staticmethod
    def diff(a: Build | dict[str, Any], b: Build | dict[str, Any]) -> DeltaResult:
        build_a = a if isinstance(a, Build) else Build.from_dict(a)
        build_b = b if isinstance(b, Build) else Build.from_dict(b)

        ids_a = set(build_a.neurons)
        ids_b = set(build_b.neurons)
        added = sorted(ids_b - ids_a)
        removed = sorted(ids_a - ids_b)
        shared = ids_a & ids_b

        splits: list[dict[str, Any]] = []
        merges: list[dict[str, Any]] = []
        for nid in removed:
            attrs_a = build_a.neurons[nid]
            matches = []
            for mid in added:
                ov = _overlap(attrs_a, build_b.neurons[mid])
                if ov >= 0.25:
                    matches.append((mid, ov))
            matches.sort(key=lambda t: -t[1])
            if len(matches) >= 2:
                splits.append(
                    {
                        "from": nid,
                        "into": [m for m, _ in matches],
                        "overlaps": {m: round(o, 6) for m, o in matches},
                    }
                )

        for nid in added:
            attrs_b = build_b.neurons[nid]
            matches = []
            for mid in removed:
                ov = _overlap(build_a.neurons[mid], attrs_b)
                if ov >= 0.25:
                    matches.append((mid, ov))
            matches.sort(key=lambda t: -t[1])
            if len(matches) >= 2:
                merges.append(
                    {
                        "into": nid,
                        "from": [m for m, _ in matches],
                        "overlaps": {m: round(o, 6) for m, o in matches},
                    }
                )

        syn_a, syn_b = set(build_a.synapses), set(build_b.synapses)
        syn_added = sorted(syn_b - syn_a)
        syn_removed = sorted(syn_a - syn_b)

        edges_a = {(pre, post) for pre, post in build_a.edges}
        edges_b = {(pre, post) for pre, post in build_b.edges}
        edges_changed = {
            "added": [list(e) for e in sorted(edges_b - edges_a)],
            "removed": [list(e) for e in sorted(edges_a - edges_b)],
        }

        comps_a = {frozenset(c) for c in build_a.components}
        comps_b = {frozenset(c) for c in build_b.components}
        component_changes = {
            "count_a": len(build_a.components),
            "count_b": len(build_b.components),
            "added": [sorted(c) for c in comps_b - comps_a],
            "removed": [sorted(c) for c in comps_a - comps_b],
        }

        degree_changes: dict[str, dict[str, int]] = {}
        for nid in sorted(set(build_a.degrees) | set(build_b.degrees)):
            da = int(build_a.degrees.get(nid, 0))
            db = int(build_b.degrees.get(nid, 0))
            if da != db:
                degree_changes[nid] = {"a": da, "b": db, "delta": db - da}

        spatial_changes: list[dict[str, Any]] = []
        confidence_changes: list[dict[str, Any]] = []
        for nid in sorted(shared):
            aa, bb = build_a.neurons[nid], build_b.neurons[nid]
            ca, cb = _centroid(aa), _centroid(bb)
            if ca is not None and cb is not None:
                d = _dist(ca, cb)
                if d > float(aa.get("spatial_tol", bb.get("spatial_tol", 1.0))):
                    spatial_changes.append(
                        {
                            "id": nid,
                            "centroid_a": list(ca),
                            "centroid_b": list(cb),
                            "distance": round(d, 6),
                        }
                    )
            conf_a = aa.get("confidence", aa.get("prediction_confidence"))
            conf_b = bb.get("confidence", bb.get("prediction_confidence"))
            if conf_a is not None and conf_b is not None and float(conf_a) != float(conf_b):
                confidence_changes.append(
                    {
                        "id": nid,
                        "confidence_a": float(conf_a),
                        "confidence_b": float(conf_b),
                        "delta": float(conf_b) - float(conf_a),
                    }
                )

        for sid in sorted(syn_a & syn_b):
            sa, sb = build_a.synapses[sid], build_b.synapses[sid]
            conf_a = sa.get("confidence", sa.get("prediction_confidence"))
            conf_b = sb.get("confidence", sb.get("prediction_confidence"))
            if conf_a is not None and conf_b is not None and float(conf_a) != float(conf_b):
                confidence_changes.append(
                    {
                        "id": sid,
                        "kind": "synapse",
                        "confidence_a": float(conf_a),
                        "confidence_b": float(conf_b),
                        "delta": float(conf_b) - float(conf_a),
                    }
                )

        return DeltaResult(
            neurons_added=added,
            neurons_removed=removed,
            objects_split=splits,
            objects_merged=merges,
            synapses_added=syn_added,
            synapses_removed=syn_removed,
            edges_changed=edges_changed,
            component_changes=component_changes,
            degree_changes=degree_changes,
            spatial_changes=spatial_changes,
            confidence_changes=confidence_changes,
        )
