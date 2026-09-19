"""Orientation-equivariant edge presentation for affinity face review.

Canonical local display frame (no privileged world XY):
  display axis 0 = affinity direction (A → B)
  display axes 1,2 = the two orthogonal world axes, oriented by a
  deterministic min-hash rule so the frame is unique under reflection

Panels (names contain no Z/Y/X):
  SIDE_A, SIDE_B: transverse planes through each endpoint
  LONGITUDINAL_1, LONGITUDINAL_2: planes containing the edge, normals along
  display-1 and display-2

Shared neighborhood intensity normalization; identical marker geometry.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

VOXEL_NM = 8.0
DEFAULT_RADIUS = 12
MARKER_HALF = 2
COLOR_A = (255, 55, 55)
COLOR_B = (60, 230, 110)


@dataclass(frozen=True)
class EdgePresentation:
    panels: dict[str, np.ndarray]
    panel_order: tuple[str, ...]
    a_display: tuple[int, int, int]
    b_display: tuple[int, int, int]
    neighborhood_shape_zyx: tuple[int, int, int]
    display_shape: tuple[int, int, int]
    radius_voxels: int
    fov_nm: float
    channel_zyx: int
    pair_left_zyx: tuple[int, int, int]
    pair_right_zyx: tuple[int, int, int]
    ortho_world_axes: tuple[int, int]
    norm_low: float
    norm_high: float


def neighborhood_origin(
    a: tuple[int, int, int], b: tuple[int, int, int], radius: int, shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    mid = tuple((ai + bi) // 2 for ai, bi in zip(a, b))
    origin = []
    size = 2 * radius + 1
    for m, lim in zip(mid, shape):
        o = int(m) - radius
        o = max(0, min(o, lim - size))
        origin.append(o)
    return tuple(origin)  # type: ignore[return-value]


def extract_cube(raw: np.ndarray, origin: tuple[int, int, int], radius: int) -> np.ndarray:
    size = 2 * radius + 1
    z0, y0, x0 = origin
    return np.asarray(raw[z0 : z0 + size, y0 : y0 + size, x0 : x0 + size], dtype=np.float32)


def _world_cube_to_display(cube: np.ndarray, channel: int, o1: int, o2: int) -> np.ndarray:
    """Remap world-local cube to display[affinity, ortho1, ortho2]."""
    size = cube.shape[0]
    # Build index grids in display coords and gather
    d0 = np.arange(size)[:, None, None]
    d1 = np.arange(size)[None, :, None]
    d2 = np.arange(size)[None, None, :]
    idx = [None, None, None]
    idx[channel] = d0
    idx[o1] = d1
    idx[o2] = d2
    return cube[idx[0], idx[1], idx[2]]


def _canonical_display(cube: np.ndarray, channel: int) -> tuple[int, int, np.ndarray, tuple[bool, bool]]:
    """Pick ortho order + optional axis flips by minimum SHA-256 of display cube.

    Affinity/display-0 direction is never flipped (A → B stays increasing depth).
    Eight candidates = {o1,o2} order × flip_d1 × flip_d2, absorbing chirality
    changes from odd world-axis permutations.
    """
    others = [i for i in (0, 1, 2) if i != channel]
    best = None
    for o1, o2 in (tuple(others), tuple(others[::-1])):
        base = _world_cube_to_display(cube, channel, o1, o2)
        for flip1 in (False, True):
            for flip2 in (False, True):
                disp = base
                if flip1:
                    disp = np.flip(disp, axis=1)
                if flip2:
                    disp = np.flip(disp, axis=2)
                key = hashlib.sha256(np.ascontiguousarray(disp).tobytes()).hexdigest()
                if best is None or key < best[0]:
                    best = (key, o1, o2, disp, (flip1, flip2))
    assert best is not None
    return best[1], best[2], best[3], best[4]


def _world_local_to_display(
    loc: tuple[int, int, int], channel: int, o1: int, o2: int, flips: tuple[bool, bool], size: int
) -> tuple[int, int, int]:
    d0 = int(loc[channel])
    d1 = int(loc[o1])
    d2 = int(loc[o2])
    if flips[0]:
        d1 = size - 1 - d1
    if flips[1]:
        d2 = size - 1 - d2
    return (d0, d1, d2)


def _to_rgb(plane: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        gray = np.zeros(plane.shape, dtype=np.uint8)
    else:
        gray = np.clip((plane - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _stamp(rgb: np.ndarray, row: int, col: int, color: tuple[int, int, int]) -> None:
    h, w = rgb.shape[:2]
    r0, r1 = max(0, row - MARKER_HALF), min(h, row + MARKER_HALF + 1)
    c0, c1 = max(0, col - MARKER_HALF), min(w, col + MARKER_HALF + 1)
    rgb[r0:r1, c0:c1] = color


def render_edge_presentation(
    raw: np.ndarray,
    pair_left_zyx: tuple[int, int, int] | list[int],
    channel_zyx: int,
    *,
    radius: int = DEFAULT_RADIUS,
) -> EdgePresentation:
    if channel_zyx not in (0, 1, 2):
        raise ValueError("channel_zyx must be 0,1,2")
    a = tuple(int(v) for v in pair_left_zyx)
    b_list = list(a)
    b_list[channel_zyx] += 1
    b = tuple(b_list)
    shape = tuple(int(s) for s in raw.shape)
    if any(v < 0 or v >= lim for v, lim in zip(a, shape)) or b[channel_zyx] >= shape[channel_zyx]:
        raise ValueError("edge outside volume")

    origin = neighborhood_origin(a, b, radius, shape)
    cube = extract_cube(raw, origin, radius)
    size = 2 * radius + 1
    if cube.shape != (size, size, size):
        raise ValueError(f"neighborhood truncated: got {cube.shape}")

    a_loc = tuple(ai - oi for ai, oi in zip(a, origin))
    b_loc = tuple(bi - oi for bi, oi in zip(b, origin))
    o1, o2, disp, flips = _canonical_display(cube, channel_zyx)
    a_d = _world_local_to_display(a_loc, channel_zyx, o1, o2, flips, size)
    b_d = _world_local_to_display(b_loc, channel_zyx, o1, o2, flips, size)
    low = float(disp.min())
    high = float(disp.max())

    def stamp_both(rgb: np.ndarray, fixed_axis: int, fixed_val: int, row_axis: int, col_axis: int) -> None:
        for pt, color in ((a_d, COLOR_A), (b_d, COLOR_B)):
            if pt[fixed_axis] != fixed_val:
                continue
            _stamp(rgb, int(pt[row_axis]), int(pt[col_axis]), color)

    # display layout: (depth=0, row=1, col=2)
    side_a = _to_rgb(disp[a_d[0], :, :], low, high)
    stamp_both(side_a, 0, a_d[0], 1, 2)
    side_b = _to_rgb(disp[b_d[0], :, :], low, high)
    stamp_both(side_b, 0, b_d[0], 1, 2)

    long1 = _to_rgb(disp[:, a_d[1], :], low, high)  # fix display-1; plane (depth, col)
    stamp_both(long1, 1, a_d[1], 0, 2)
    long2 = _to_rgb(disp[:, :, a_d[2]], low, high)  # fix display-2; plane (depth, row)
    stamp_both(long2, 2, a_d[2], 0, 1)

    panel_order = ("SIDE_A", "SIDE_B", "LONGITUDINAL_1", "LONGITUDINAL_2")
    panels = {
        "SIDE_A": side_a,
        "SIDE_B": side_b,
        "LONGITUDINAL_1": long1,
        "LONGITUDINAL_2": long2,
    }
    return EdgePresentation(
        panels=panels,
        panel_order=panel_order,
        a_display=a_d,
        b_display=b_d,
        neighborhood_shape_zyx=tuple(int(s) for s in cube.shape),
        display_shape=tuple(int(s) for s in disp.shape),
        radius_voxels=radius,
        fov_nm=float(size * VOXEL_NM),
        channel_zyx=channel_zyx,
        pair_left_zyx=a,
        pair_right_zyx=b,
        ortho_world_axes=(o1, o2),
        norm_low=low,
        norm_high=high,
    )


def presentation_fingerprint(pres: EdgePresentation) -> dict[str, str]:
    out = {}
    for name in pres.panel_order:
        out[name] = hashlib.sha256(np.ascontiguousarray(pres.panels[name]).tobytes()).hexdigest()
    return out


def both_markers_visible(pres: EdgePresentation) -> dict[str, dict[str, bool]]:
    def has(panel: np.ndarray, color: tuple[int, int, int]) -> bool:
        return bool(np.any(np.all(panel == np.asarray(color, dtype=np.uint8), axis=-1)))

    return {
        name: {"A": has(pres.panels[name], COLOR_A), "B": has(pres.panels[name], COLOR_B)}
        for name in pres.panel_order
    }
