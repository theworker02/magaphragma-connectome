"""Geometry + redundancy audit for S7 / HyperDrain tiling."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


# Frozen FAST_001/002 production tile contract (measured 3468 tiles on 128³×1024² cores).
CROP_ZYX = (20, 64, 64)
STRIDE_ZYX = (10, 64, 64)
CORE_SHAPE_ZYX = (128, 1024, 1024)

# Spec mentions ~20,736 tiles — that figure is NOT the current FAST core geometry.
# Documented here so audits do not silently invent it.
SPEC_TILE_HINT = 20736


@dataclass(frozen=True)
class TileLayout:
    volume_zyx: tuple[int, int, int]
    crop_zyx: tuple[int, int, int]
    stride_zyx: tuple[int, int, int]
    padding: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    counts_zyx: tuple[int, int, int]
    padded_zyx: tuple[int, int, int]
    n_tiles: int


def tile_layout(
    volume_zyx: tuple[int, int, int] = CORE_SHAPE_ZYX,
    crop_zyx: tuple[int, int, int] = CROP_ZYX,
    stride_zyx: tuple[int, int, int] = STRIDE_ZYX,
) -> TileLayout:
    """Match run_affinity_fullvol_s7_fast_worker.tile_layout exactly."""
    counts = tuple(max(1, (n - c) // s + 2) for n, c, s in zip(volume_zyx, crop_zyx, stride_zyx))
    padded = tuple(c + (count - 1) * s for c, count, s in zip(crop_zyx, counts, stride_zyx))
    padding = tuple(((p - n) // 2, (p - n + 1) // 2) for n, p in zip(volume_zyx, padded))
    n_tiles = int(counts[0] * counts[1] * counts[2])
    return TileLayout(
        volume_zyx=volume_zyx,
        crop_zyx=crop_zyx,
        stride_zyx=stride_zyx,
        padding=padding,
        counts_zyx=counts,  # type: ignore[arg-type]
        padded_zyx=padded,  # type: ignore[arg-type]
        n_tiles=n_tiles,
    )


def axis_coverage(length: int, crop: int, stride: int) -> list[int]:
    """How many tiles cover each voxel along one axis (unpadded volume indexing approx via starts)."""
    # Approximate using the production start grid after padding crop-aligned.
    # For redundancy we count overlaps on the *padded* axis then map conceptually.
    counts = max(1, (length - crop) // stride + 2)
    padded = crop + (counts - 1) * stride
    pad0 = (padded - length) // 2
    cover = [0] * length
    for i in range(counts):
        start = i * stride - pad0
        for v in range(max(0, start), min(length, start + crop)):
            cover[v] += 1
    return cover


def receptive_field_estimate() -> dict:
    """
    MNet is a multi-resolution U-Net with same-padded 3×3 (and 1×3×3) convs and
    4 levels of 2× pooling on XY (and Z when deep enough).

    Same-padding ⇒ output spatial size equals input; every output voxel is a
    function of a large (effectively near-global within the tile) context.
    Production S7 does NOT hard-crop a valid center; it Gaussian-blends full
    tile predictions. Therefore the practical 'halo' for stitching dense macros
    is the blend transition, not a classical valid-conv crop.

    Conservative engineering halo for macro-tile borders (to discard when
    stitching without blend): half the smaller of (crop_z, crop_y) ≈ 10/32.
    We use (4, 8, 8) as a *safety discard* when comparing dense vs tiled blend
    only if dense path uses hard crop; default dense path still uses the same
    Gaussian blend contract as eager tiles.
    """
    return {
        "network": "MNet kn=(32,64,96,128,256) FMU=sub (production ckpt)",
        "padding_mode": "same (output size == input size per forward)",
        "pooling_levels_xy": 4,
        "conv_kernels": ["3x3x3", "1x3x3"],
        "theoretical_rf_note": (
            "With 4× pool and stacked 3×3, theoretical RF exceeds the 20×64×64 tile; "
            "within-tile context is effectively the full tile."
        ),
        "production_stitch": "gaussian_blend_full_tile",
        "macro_border_discard_zyx_if_hard_crop": [4, 8, 8],
        "halo_for_blend_contract": [0, 0, 0],
    }


def redundancy_audit(
    volume_zyx: tuple[int, int, int] = CORE_SHAPE_ZYX,
    crop_zyx: tuple[int, int, int] = CROP_ZYX,
    stride_zyx: tuple[int, int, int] = STRIDE_ZYX,
) -> dict:
    layout = tile_layout(volume_zyx, crop_zyx, stride_zyx)
    covers = [axis_coverage(n, c, s) for n, c, s in zip(volume_zyx, crop_zyx, stride_zyx)]
    mean_cov = [sum(c) / max(1, len(c)) for c in covers]
    # Voxel-eval multiplier ≈ product of per-axis mean coverage for separable grids.
    redundancy_factor = 1.0
    for m in mean_cov:
        redundancy_factor *= m

    tile_voxels = crop_zyx[0] * crop_zyx[1] * crop_zyx[2]
    volume_voxels = volume_zyx[0] * volume_zyx[1] * volume_zyx[2]
    total_input_voxels_evaluated = layout.n_tiles * tile_voxels

    # Adjacent-tile overlap fractions (1 - stride/crop) when stride < crop, else 0.
    adj_overlap = []
    for c, s in zip(crop_zyx, stride_zyx):
        adj_overlap.append(max(0.0, 1.0 - (s / c)))

    # Adjacent chunk: cores are abutting 1024²×128 without shared compute in FAST core-only mode.
    adjacent_chunk_overlap = {
        "fast_core_only": True,
        "xy_shared_voxels": 0,
        "z_shared_voxels": 0,
        "note": "FAST contracts read core_bounds only; no cross-chunk tile overlap in compute.",
    }

    dense_forwards_if_one_shot = 1
    # Macro example: cover volume with non-overlapping macros of size crop* k …
    return {
        "schema_version": 1,
        "geometry": asdict(layout),
        "spec_tile_hint_20736": {
            "value": SPEC_TILE_HINT,
            "matches_current_fast_core": SPEC_TILE_HINT == layout.n_tiles,
            "note": (
                "Current FAST core (128,1024,1024)+(20,64,64)@(10,64,64) yields "
                f"{layout.n_tiles} tiles, not 20736. Audit uses measured FAST geometry."
            ),
        },
        "receptive_field": receptive_field_estimate(),
        "per_axis_mean_coverage": {
            "z": mean_cov[0],
            "y": mean_cov[1],
            "x": mean_cov[2],
        },
        "adjacent_tile_overlap_fraction_zyx": adj_overlap,
        "adjacent_chunk_overlap": adjacent_chunk_overlap,
        "volume_voxels": volume_voxels,
        "tile_voxels": tile_voxels,
        "n_tiles": layout.n_tiles,
        "total_input_voxels_evaluated": total_input_voxels_evaluated,
        "redundancy_before": redundancy_factor,
        "redundancy_interpretation": (
            "Mean number of tile forwards covering each output voxel (separable grid approx)."
        ),
        "theoretical_dense_forwards_lower_bound": dense_forwards_if_one_shot,
        "effective_compute_reduction_if_zero_overlap": redundancy_factor / dense_forwards_if_one_shot,
        "warrant_macro_tile": redundancy_factor >= 1.5 or adj_overlap[0] > 0.0,
        "recommendation": (
            "Z-stride is half crop → ~2× Z overlap. Prefer Z-elongated macro-tiles "
            "(e.g. 40×64×64 or larger XY macros with stride=crop) plus Gaussian blend "
            "to cut tile-launch overhead; full-volume single forward exceeds VRAM."
        ),
    }


def write_audit(path: Path, **kwargs) -> dict:
    audit = redundancy_audit(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def starts_for_axis(count: int, stride: int) -> tuple[int, ...]:
    return tuple(i * stride for i in range(count))


def iter_tile_positions(layout: TileLayout) -> Iterable[tuple[int, int, int]]:
    zs = starts_for_axis(layout.counts_zyx[0], layout.stride_zyx[0])
    ys = starts_for_axis(layout.counts_zyx[1], layout.stride_zyx[1])
    xs = starts_for_axis(layout.counts_zyx[2], layout.stride_zyx[2])
    for z in zs:
        for y in ys:
            for x in xs:
                yield (z, y, x)
