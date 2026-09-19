"""MorphGuard morphology auditor.

Detects heuristic morphology flags on label volumes or skeleton graphs.
Never claims definitive biology error ? outputs AnomalyMap + review queue.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from morphguard.anomalies import Anomaly, AnomalyMap

try:
    from scipy import ndimage as ndi

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    ndi = None  # type: ignore
    _HAS_SCIPY = False


@dataclass
class SkeletonGraph:
    """Undirected skeleton: nodes (N,3), edges (E,2) int indices."""

    nodes: np.ndarray
    edges: np.ndarray
    label_id: int | str = 0

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=np.float64)
        self.edges = np.asarray(self.edges, dtype=np.int64)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must be (N,3)")
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges must be (E,2)")


def _label_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    if _HAS_SCIPY:
        labeled, n = ndi.label(binary)
        return labeled, int(n)
    labeled = np.zeros(binary.shape, dtype=np.int32)
    n = 0
    coords = list(zip(*np.nonzero(binary)))
    visited = set()
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    shape = binary.shape
    for seed in coords:
        if seed in visited:
            continue
        n += 1
        stack = [seed]
        visited.add(seed)
        while stack:
            z, y, x = stack.pop()
            labeled[z, y, x] = n
            for dz, dy, dx in neighbors:
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                    key = (nz, ny, nx)
                    if binary[nz, ny, nx] and key not in visited:
                        visited.add(key)
                        stack.append(key)
    return labeled, n


def _distance_transform(binary: np.ndarray) -> np.ndarray:
    """Local thickness proxy: distance to background."""
    if _HAS_SCIPY:
        return ndi.distance_transform_edt(binary)
    dist = np.zeros(binary.shape, dtype=np.float64)
    cur = binary.astype(bool).copy()
    layer = 0
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while cur.any():
        layer += 1
        dist[cur] = layer
        nxt = np.zeros_like(cur)
        zz, yy, xx = np.nonzero(cur)
        for z, y, x in zip(zz.tolist(), yy.tolist(), xx.tolist()):
            keep = True
            for dz, dy, dx in neighbors:
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < cur.shape[0] and 0 <= ny < cur.shape[1] and 0 <= nx < cur.shape[2]:
                    if not cur[nz, ny, nx]:
                        keep = False
                        break
            if keep:
                nxt[z, y, x] = True
        cur = nxt
        if layer > max(binary.shape):
            break
    return dist


def _touches_boundary(mask: np.ndarray) -> bool:
    z, y, x = mask.shape
    return bool(
        mask[0].any()
        or mask[z - 1].any()
        or mask[:, 0].any()
        or mask[:, y - 1].any()
        or mask[:, :, 0].any()
        or mask[:, :, x - 1].any()
    )


def _thin_neck_at_boundary(mask: np.ndarray, max_thickness: float = 1.5) -> bool:
    if not _touches_boundary(mask):
        return False
    dist = _distance_transform(mask)
    z, y, x = mask.shape
    faces = [
        dist[0][mask[0]],
        dist[z - 1][mask[z - 1]],
        dist[:, 0][mask[:, 0]],
        dist[:, y - 1][mask[:, y - 1]],
        dist[:, :, 0][mask[:, :, 0]],
        dist[:, :, x - 1][mask[:, :, x - 1]],
    ]
    vals = np.concatenate([f.ravel() for f in faces if f.size]) if any(f.size for f in faces) else np.array([])
    if vals.size == 0:
        return False
    return float(np.min(vals)) <= max_thickness


def audit_label_volume(
    labels: np.ndarray,
    *,
    tiny_threshold: int = 20,
    diameter_jump_ratio: float = 3.0,
    branch_degree_threshold: int = 4,
) -> AnomalyMap:
    labs = np.asarray(labels)
    if labs.ndim != 3:
        raise ValueError("labels must be 3D")
    amap = AnomalyMap(
        meta={
            "input": "label_volume",
            "shape": list(labs.shape),
            "scipy": _HAS_SCIPY,
            "tiny_threshold": tiny_threshold,
            "disclaimer": "Heuristic flags only",
        }
    )
    ids = [int(i) for i in np.unique(labs) if int(i) != 0]
    for lid in sorted(ids):
        mask = labs == lid
        count = int(mask.sum())
        sid = str(lid)

        if count < tiny_threshold:
            amap.add(
                Anomaly(
                    kind="tiny_fragment",
                    subject_id=sid,
                    score=min(1.0, (tiny_threshold - count) / max(1, tiny_threshold)),
                    evidence={"voxels": count, "threshold": tiny_threshold},
                )
            )

        labeled, ncomp = _label_components(mask)
        if ncomp > 1:
            sizes = [int(np.sum(labeled == c)) for c in range(1, ncomp + 1)]
            amap.add(
                Anomaly(
                    kind="discontinuity",
                    subject_id=sid,
                    score=min(1.0, 0.4 + 0.15 * (ncomp - 1)),
                    evidence={"n_components": ncomp, "component_sizes": sizes},
                )
            )
            if sizes and max(sizes) >= tiny_threshold:
                for i, s in enumerate(sizes):
                    if s < tiny_threshold:
                        amap.add(
                            Anomaly(
                                kind="orphan_process",
                                subject_id=f"{sid}#comp{i+1}",
                                score=min(1.0, 0.5 + (tiny_threshold - s) / max(1, tiny_threshold)),
                                evidence={"voxels": s, "parent_label": lid},
                            )
                        )

        if count >= 8:
            dist = _distance_transform(mask)
            means = []
            for zi in range(labs.shape[0]):
                sl = dist[zi][mask[zi]]
                if sl.size:
                    means.append(float(np.mean(sl)))
            if len(means) >= 2:
                arr = np.asarray(means, dtype=np.float64)
                prev = np.maximum(arr[:-1], 1e-6)
                ratios = arr[1:] / prev
                max_ratio = float(np.max(np.maximum(ratios, 1.0 / ratios)))
                if max_ratio >= diameter_jump_ratio:
                    amap.add(
                        Anomaly(
                            kind="abrupt_diameter_change",
                            subject_id=sid,
                            score=min(1.0, (max_ratio - 1.0) / max(1.0, diameter_jump_ratio)),
                            evidence={"max_ratio": round(max_ratio, 4), "threshold": diameter_jump_ratio},
                        )
                    )

        if _thin_neck_at_boundary(mask):
            amap.add(
                Anomaly(
                    kind="boundary_break",
                    subject_id=sid,
                    score=0.55,
                    evidence={"touches_volume_edge": True, "thin_neck": True},
                )
            )

        if count >= 30 and _HAS_SCIPY:
            struct = ndi.generate_binary_structure(3, 1)
            eroded = ndi.binary_erosion(mask, structure=struct, iterations=1)
            if eroded.any():
                neigh = ndi.convolve(eroded.astype(np.int16), struct.astype(np.int16), mode="constant")
                branch = (eroded) & (neigh >= (branch_degree_threshold + 1))
                if branch.any():
                    dist = _distance_transform(mask)
                    bvals = dist[branch]
                    mean_t = float(np.mean(dist[mask])) + 1e-6
                    mismatch = float(np.max(np.abs(bvals - mean_t) / mean_t))
                    if mismatch >= 0.8:
                        amap.add(
                            Anomaly(
                                kind="implausible_merge_point",
                                subject_id=sid,
                                score=min(1.0, 0.4 + 0.4 * mismatch),
                                evidence={
                                    "n_branch_voxels": int(branch.sum()),
                                    "thickness_mismatch": round(mismatch, 4),
                                },
                            )
                        )

    return amap


def audit_skeletons(
    graphs: Iterable[SkeletonGraph],
    *,
    edge_outlier_z: float = 3.0,
    branch_degree_threshold: int = 4,
    tortuosity_threshold: float = 3.5,
) -> AnomalyMap:
    amap = AnomalyMap(
        meta={
            "input": "skeleton_graphs",
            "scipy": _HAS_SCIPY,
            "disclaimer": "Heuristic flags only",
        }
    )
    for g in graphs:
        sid = str(g.label_id)
        nodes = g.nodes
        edges = g.edges
        if edges.size == 0:
            continue

        lengths = []
        for i, j in edges:
            lengths.append(float(np.linalg.norm(nodes[int(i)] - nodes[int(j)])))
        lengths_arr = np.asarray(lengths, dtype=np.float64)
        mu = float(np.mean(lengths_arr))
        sigma = float(np.std(lengths_arr)) + 1e-9
        for idx, L in enumerate(lengths_arr.tolist()):
            z = (L - mu) / sigma
            if z >= edge_outlier_z:
                amap.add(
                    Anomaly(
                        kind="spatial_jump",
                        subject_id=f"{sid}#e{idx}",
                        score=min(1.0, z / (edge_outlier_z + 2.0)),
                        evidence={"length": round(L, 4), "z_score": round(z, 4)},
                    )
                )

        deg: dict[int, int] = defaultdict(int)
        undirected: set[tuple[int, int]] = set()
        parallel = 0
        for i, j in edges:
            a, b = int(i), int(j)
            deg[a] += 1
            deg[b] += 1
            key = (min(a, b), max(a, b))
            if key in undirected:
                parallel += 1
            else:
                undirected.add(key)
        if parallel:
            amap.add(
                Anomaly(
                    kind="duplicate_branch",
                    subject_id=sid,
                    score=min(1.0, 0.5 + 0.1 * parallel),
                    evidence={"parallel_edges": parallel},
                )
            )
        for n, d in sorted(deg.items()):
            if d > branch_degree_threshold:
                amap.add(
                    Anomaly(
                        kind="suspicious_branching",
                        subject_id=f"{sid}#n{n}",
                        score=min(1.0, 0.4 + 0.1 * (d - branch_degree_threshold)),
                        evidence={"degree": d, "threshold": branch_degree_threshold},
                    )
                )

        leaves = [n for n, d in deg.items() if d == 1]
        if len(leaves) >= 2 and undirected:
            adj: dict[int, list[int]] = defaultdict(list)
            for a, b in undirected:
                adj[a].append(b)
                adj[b].append(a)

            def path_stats(start: int, end: int):
                parent = {start: -1}
                q = [start]
                while q:
                    cur = q.pop(0)
                    if cur == end:
                        break
                    for nxt in adj[cur]:
                        if nxt not in parent:
                            parent[nxt] = cur
                            q.append(nxt)
                if end not in parent:
                    return None
                path = [end]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                plen = 0.0
                for u, v in zip(path[:-1], path[1:]):
                    plen += float(np.linalg.norm(nodes[u] - nodes[v]))
                e2e = float(np.linalg.norm(nodes[start] - nodes[end])) + 1e-9
                return plen, e2e

            best = 0.0
            best_ev: dict[str, Any] = {}
            leaf_list = sorted(leaves)[:12]
            for i in range(len(leaf_list)):
                for j in range(i + 1, len(leaf_list)):
                    stats = path_stats(leaf_list[i], leaf_list[j])
                    if not stats:
                        continue
                    plen, e2e = stats
                    tort = plen / e2e
                    if tort > best:
                        best = tort
                        best_ev = {
                            "tortuosity": round(tort, 4),
                            "path_length": round(plen, 4),
                            "end_to_end": round(e2e, 4),
                            "leaf_a": leaf_list[i],
                            "leaf_b": leaf_list[j],
                        }
            if best >= tortuosity_threshold:
                amap.add(
                    Anomaly(
                        kind="extreme_tortuosity",
                        subject_id=sid,
                        score=min(1.0, best / (tortuosity_threshold + 2.0)),
                        evidence=best_ev,
                    )
                )

            for n, d in deg.items():
                if d == 3:
                    lens = []
                    for a, b in undirected:
                        if a == n or b == n:
                            other = b if a == n else a
                            lens.append(float(np.linalg.norm(nodes[n] - nodes[other])))
                    if len(lens) >= 3:
                        mx, mn = max(lens), min(lens) + 1e-9
                        if mx / mn >= 4.0:
                            amap.add(
                                Anomaly(
                                    kind="implausible_merge_point",
                                    subject_id=f"{sid}#n{n}",
                                    score=min(1.0, 0.35 + 0.1 * (mx / mn)),
                                    evidence={"incident_lengths": [round(x, 4) for x in lens]},
                                )
                            )

    return amap


class MorphGuard:
    """Morphology auditor entrypoint."""

    def analyze(
        self,
        labels: np.ndarray | None = None,
        skeletons: list[SkeletonGraph] | None = None,
        **kwargs: Any,
    ) -> AnomalyMap:
        maps: list[AnomalyMap] = []
        if labels is not None:
            kw = {k: v for k, v in kwargs.items() if k in {
                "tiny_threshold", "diameter_jump_ratio", "branch_degree_threshold"
            }}
            maps.append(audit_label_volume(labels, **kw))
        if skeletons is not None:
            kw = {k: v for k, v in kwargs.items() if k in {
                "edge_outlier_z", "branch_degree_threshold", "tortuosity_threshold"
            }}
            maps.append(audit_skeletons(skeletons, **kw))
        if not maps:
            raise ValueError("provide labels and/or skeletons")
        if len(maps) == 1:
            return maps[0]
        merged = AnomalyMap(meta={"input": "combined", "disclaimer": "Heuristic flags only"})
        for m in maps:
            for a in m.anomalies:
                merged.add(a)
            merged.meta.update({f"sub_{k}": v for k, v in m.meta.items()})
        return merged
