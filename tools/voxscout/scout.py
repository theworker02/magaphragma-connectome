"""VoxScout ? intelligent spatial planner for connectomics volumes.

Classification thresholds (applied in priority order; every tile kept):

  ARTIFACT          artifact_score >= 0.12
  EMPTY             tissue_probability < 0.05 AND information_density < 0.8
  UNCERTAIN         uncertainty >= 0.75
  REPROCESS         prior_coverage >= 0.5 AND uncertainty >= 0.45
  HIGH_COMPLEXITY   boundary_complexity >= 0.25 OR information_density >= 4.5
  LOW_INFORMATION   information_density < 1.2 AND boundary_complexity < 0.08
                    AND tissue_probability < 0.35
  NORMAL            otherwise

recommend_priority in [0, 1]: higher means process sooner / more carefully.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from voxscout._core_compat import load_core
from voxscout.priority_map import PriorityMap, TilePriority
from voxscout.__version__ import __version__

RegionClass, TileIndex, write_tool_receipt, ArtifactStore, ProvenanceGraph = load_core()


def _shannon_entropy_hist(hist: np.ndarray) -> float:
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0
    p = hist.astype(np.float64) / total
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p)))


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError("volume must be 3D (Z,Y,X)")
    if np.issubdtype(vol.dtype, np.floating):
        v = vol.astype(np.float64)
        if v.max() <= 1.0 + 1e-9:
            v = v * 255.0
        return np.clip(v, 0.0, 255.0)
    return vol.astype(np.float64)


def _tissue_probability(tile: np.ndarray) -> float:
    hist, _ = np.histogram(tile.ravel(), bins=256, range=(0.0, 255.0))
    mid = float(hist[64:193].sum())
    total = float(hist.sum()) + 1e-12
    return mid / total


def _information_density(tile: np.ndarray) -> float:
    hist, _ = np.histogram(tile.ravel(), bins=256, range=(0.0, 255.0))
    return _shannon_entropy_hist(hist)


def _boundary_complexity(tile: np.ndarray) -> float:
    gz, gy, gx = np.gradient(tile.astype(np.float64))
    mag = np.sqrt(gz * gz + gy * gy + gx * gx)
    return float(np.mean(mag) / 255.0)


def _artifact_score(tile: np.ndarray) -> float:
    flat = tile.ravel()
    n = max(1, flat.size)
    lo = float(np.sum(flat <= 2.0)) / n
    hi = float(np.sum(flat >= 253.0)) / n
    sat = lo + hi
    if tile.size >= 27:
        pad = np.pad(tile, 1, mode="edge")
        center = pad[1:-1, 1:-1, 1:-1]
        neigh = (
            pad[:-2, 1:-1, 1:-1]
            + pad[2:, 1:-1, 1:-1]
            + pad[1:-1, :-2, 1:-1]
            + pad[1:-1, 2:, 1:-1]
            + pad[1:-1, 1:-1, :-2]
            + pad[1:-1, 1:-1, 2:]
        ) / 6.0
        impulse = float(np.mean(np.abs(center - neigh) > 80.0))
    else:
        impulse = 0.0
    return float(min(1.0, 0.55 * sat + 0.45 * impulse))


def _uncertainty(tile: np.ndarray) -> float:
    z, y, x = tile.shape
    zs = [slice(0, max(1, z // 2)), slice(max(1, z // 2), z)]
    ys = [slice(0, max(1, y // 2)), slice(max(1, y // 2), y)]
    xs = [slice(0, max(1, x // 2)), slice(max(1, x // 2), x)]
    means = []
    for za in zs:
        for ya in ys:
            for xa in xs:
                block = tile[za, ya, xa]
                means.append(float(np.mean(block)) if block.size else 0.0)
    means_arr = np.asarray(means, dtype=np.float64)
    bins = np.clip((means_arr / 255.0 * 8.0).astype(np.int64), 0, 7)
    hist = np.bincount(bins, minlength=8).astype(np.float64)
    ent = _shannon_entropy_hist(hist)
    return float(min(1.0, ent / 3.0))


def _classify(tissue, info, boundary, artifact, unc, prior) -> Any:
    if artifact >= 0.12:
        return RegionClass.ARTIFACT
    if tissue < 0.05 and info < 0.8:
        return RegionClass.EMPTY
    if unc >= 0.75:
        return RegionClass.UNCERTAIN
    if prior >= 0.5 and unc >= 0.45:
        return RegionClass.REPROCESS
    if boundary >= 0.25 or info >= 4.5:
        return RegionClass.HIGH_COMPLEXITY
    if info < 1.2 and boundary < 0.08 and tissue < 0.35:
        return RegionClass.LOW_INFORMATION
    return RegionClass.NORMAL


def _recommend_priority(cls, tissue, info, boundary, artifact, unc, prior) -> float:
    base = (
        0.30 * unc
        + 0.25 * artifact
        + 0.20 * min(1.0, boundary / 0.5)
        + 0.15 * (1.0 - prior)
        + 0.10 * min(1.0, info / 8.0)
    )
    boost = {
        "ARTIFACT": 0.25,
        "UNCERTAIN": 0.22,
        "REPROCESS": 0.18,
        "HIGH_COMPLEXITY": 0.15,
        "NORMAL": 0.05,
        "LOW_INFORMATION": 0.02,
        "EMPTY": 0.0,
    }
    name = cls.value if hasattr(cls, "value") else str(cls)
    return float(min(1.0, max(0.0, base + boost.get(name, 0.05))))


class VoxScout:
    """Analyze EM volumes into a complete PriorityMap."""

    def __init__(self, tile_zyx: tuple[int, int, int] = (32, 64, 64)) -> None:
        self.tile_zyx = tile_zyx

    def analyze(self, volume: np.ndarray, prior_mask: np.ndarray | None = None) -> PriorityMap:
        vol = _normalize_volume(volume)
        if prior_mask is not None and prior_mask.shape != vol.shape:
            raise ValueError("prior_mask must match volume shape")

        index = TileIndex(shape_zyx=tuple(int(s) for s in vol.shape), tile_zyx=self.tile_zyx)
        tiles: list[TilePriority] = []

        for key, box in index.iter_tiles():
            tile = index.crop(vol, box)
            mask_tile = index.crop(prior_mask, box) if prior_mask is not None else None
            tissue = _tissue_probability(tile)
            info = _information_density(tile)
            boundary = _boundary_complexity(tile)
            artifact = _artifact_score(tile)
            unc = _uncertainty(tile)
            prior = float(np.mean(mask_tile > 0)) if mask_tile is not None else 0.0
            cls = _classify(tissue, info, boundary, artifact, unc, prior)
            pri = _recommend_priority(cls, tissue, info, boundary, artifact, unc, prior)
            cls_name = cls.value if hasattr(cls, "value") else str(cls)
            box_dict = box.as_dict() if hasattr(box, "as_dict") else {
                "z0": box.z0, "y0": box.y0, "x0": box.x0, "z1": box.z1, "y1": box.y1, "x1": box.x1
            }
            tiles.append(
                TilePriority(
                    tile_key=key,
                    region_class=cls_name,
                    tissue_probability=round(tissue, 6),
                    information_density=round(info, 6),
                    boundary_complexity=round(boundary, 6),
                    artifact_score=round(artifact, 6),
                    uncertainty=round(unc, 6),
                    prior_coverage=round(prior, 6),
                    recommend_priority=round(pri, 6),
                    box=box_dict,
                )
            )

        tiles.sort(key=lambda t: (-t.recommend_priority, t.tile_key))
        return PriorityMap(
            volume_shape=tuple(int(s) for s in vol.shape),
            tile_zyx=self.tile_zyx,
            tiles=tiles,
            tool_version=__version__,
        )
