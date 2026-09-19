"""Seam representations ? chunk A/B label volumes with overlap halo."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Face = Literal["z_pos", "z_neg", "y_pos", "y_neg", "x_pos", "x_neg"]


@dataclass
class SeamChunk:
    """Labeled chunk with optional world offset (Z,Y,X)."""

    labels: np.ndarray
    origin_zyx: tuple[int, int, int] = (0, 0, 0)
    name: str = "chunk"

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels)
        if self.labels.ndim != 3:
            raise ValueError("labels must be 3D")


@dataclass
class FaceObject:
    chunk: str
    label_id: int
    face: Face
    centroid_zyx: tuple[float, float, float]
    volume: int
    bbox_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    major_axis: tuple[float, float, float]
    voxel_coords: np.ndarray


@dataclass
class SeamPair:
    """Two chunks sharing an overlap halo along one axis."""

    chunk_a: SeamChunk
    chunk_b: SeamChunk
    axis: Literal["z", "y", "x"]
    halo: int
    a_face: Face = "x_pos"
    b_face: Face = "x_neg"


def _pca_major_axis(coords: np.ndarray) -> tuple[float, float, float]:
    if coords.shape[0] < 2:
        return (1.0, 0.0, 0.0)
    c = coords.astype(np.float64)
    c = c - c.mean(axis=0, keepdims=True)
    cov = (c.T @ c) / max(1, c.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    n = float(np.linalg.norm(axis)) + 1e-12
    axis = axis / n
    for v in axis:
        if abs(v) > 1e-12:
            if v < 0:
                axis = -axis
            break
    return (float(axis[0]), float(axis[1]), float(axis[2]))


def _bbox(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    return (
        (int(mn[0]), int(mx[0])),
        (int(mn[1]), int(mx[1])),
        (int(mn[2]), int(mx[2])),
    )


def _face_slices(shape: tuple[int, int, int], face: Face, halo: int) -> tuple[slice, slice, slice]:
    z, y, x = shape
    h = max(1, int(halo))
    if face == "z_pos":
        return (slice(max(0, z - h), z), slice(0, y), slice(0, x))
    if face == "z_neg":
        return (slice(0, min(h, z)), slice(0, y), slice(0, x))
    if face == "y_pos":
        return (slice(0, z), slice(max(0, y - h), y), slice(0, x))
    if face == "y_neg":
        return (slice(0, z), slice(0, min(h, y)), slice(0, x))
    if face == "x_pos":
        return (slice(0, z), slice(0, y), slice(max(0, x - h), x))
    if face == "x_neg":
        return (slice(0, z), slice(0, y), slice(0, min(h, x)))
    raise ValueError(f"unknown face {face}")


def extract_face_objects(chunk: SeamChunk, face: Face, halo: int) -> list[FaceObject]:
    """Extract objects that touch the opposing face within the halo slab."""
    labs = chunk.labels
    sl = _face_slices(labs.shape, face, halo)
    slab = labs[sl]
    ids = [int(i) for i in np.unique(slab) if int(i) != 0]
    out: list[FaceObject] = []
    z0 = sl[0].start or 0
    y0 = sl[1].start or 0
    x0 = sl[2].start or 0
    for lid in sorted(ids):
        mask = slab == lid
        coords_local = np.column_stack(np.nonzero(mask))
        coords = coords_local.copy()
        coords[:, 0] += z0
        coords[:, 1] += y0
        coords[:, 2] += x0
        cent = coords.mean(axis=0)
        world = (
            float(cent[0] + chunk.origin_zyx[0]),
            float(cent[1] + chunk.origin_zyx[1]),
            float(cent[2] + chunk.origin_zyx[2]),
        )
        vol = int(np.sum(labs == lid))
        full_coords = np.column_stack(np.nonzero(labs == lid))
        out.append(
            FaceObject(
                chunk=chunk.name,
                label_id=lid,
                face=face,
                centroid_zyx=world,
                volume=vol,
                bbox_zyx=_bbox(full_coords),
                major_axis=_pca_major_axis(full_coords),
                voxel_coords=coords,
            )
        )
    return out


def build_axis_seam(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    *,
    axis: Literal["z", "y", "x"] = "x",
    halo: int = 4,
    origin_a: tuple[int, int, int] = (0, 0, 0),
    origin_b: tuple[int, int, int] | None = None,
) -> SeamPair:
    """Build a seam where B is placed adjacent to A's positive face along axis."""
    a = SeamChunk(labels_a, origin_zyx=origin_a, name="A")
    if origin_b is None:
        sa = labels_a.shape
        if axis == "z":
            origin_b = (origin_a[0] + sa[0] - halo, origin_a[1], origin_a[2])
        elif axis == "y":
            origin_b = (origin_a[0], origin_a[1] + sa[1] - halo, origin_a[2])
        else:
            origin_b = (origin_a[0], origin_a[1], origin_a[2] + sa[2] - halo)
    b = SeamChunk(labels_b, origin_zyx=origin_b, name="B")
    faces = {
        "z": ("z_pos", "z_neg"),
        "y": ("y_pos", "y_neg"),
        "x": ("x_pos", "x_neg"),
    }
    af, bf = faces[axis]
    return SeamPair(chunk_a=a, chunk_b=b, axis=axis, halo=halo, a_face=af, b_face=bf)  # type: ignore[arg-type]


@dataclass
class SeamReport:
    candidates: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"candidates": self.candidates, "meta": self.meta, "n_candidates": len(self.candidates)}
