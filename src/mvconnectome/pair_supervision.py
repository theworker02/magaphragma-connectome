"""Conservative three-state DVID affinity-pair pseudo-supervision."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

from .io import sha256_file, write_json_atomic

OFFSETS_ZYX = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
UTC = timezone.utc


def _native_path(value: str) -> Path:
    """Translate recorded Windows paths only when executing under WSL."""
    match = re.match(r"^([A-Za-z]):\\(.*)$", value)
    if match and Path("/mnt").exists():
        return Path("/mnt") / match.group(1).lower() / match.group(2).replace("\\", "/")
    return Path(value)


def _slices(shape: tuple[int, int, int], offset: tuple[int, int, int]) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Return aligned neighbor slices while preventing NumPy edge wraparound."""
    left, right = [], []
    for size, delta in zip(shape, offset, strict=True):
        left.append(slice(-delta, size) if delta < 0 else slice(0, size - delta))
        right.append(slice(0, size + delta) if delta < 0 else slice(delta, size))
    return tuple(left), tuple(right)


def _supervoxels(boundary: np.ndarray, cut: float, seed_distance: int) -> np.ndarray:
    """Produce local machine fragments used only to form conservative pair targets."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    interior = boundary < cut
    distance = ndimage.distance_transform_edt(interior)
    points = peak_local_max(distance, min_distance=seed_distance, labels=interior, exclude_border=2)
    markers = np.zeros(boundary.shape, dtype=np.int32)
    if len(points):
        markers[tuple(points.T)] = np.arange(1, len(points) + 1)
    return watershed(boundary, markers=markers, mask=interior).astype(np.uint32)


def build_pair_supervision(crop_manifest_path: Path, crop_id: str, affinity_path: Path, output_dir: Path) -> dict:
    """Build three-state SAME/DIFFERENT/IGNORE affinity supervision for one crop."""
    manifest = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
    crop = next((item for item in manifest["crops"] if item["id"] == crop_id), None)
    if crop is None or crop["parent_region_id"] == "MV-GTVOL-000004":
        raise ValueError("Unknown or regression-only crop")
    raw = np.load(_native_path(crop["raw_crop_path"]), allow_pickle=False)
    affinity = np.load(affinity_path, allow_pickle=False)
    if affinity.shape != (3,) + raw.shape:
        raise ValueError("Affinity must be aligned CZYX")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    gradient = ndimage.gaussian_gradient_magnitude(raw.astype("float32"), sigma=1.0)
    boundary = np.clip(gradient / np.percentile(gradient, 99.5), 0, 1).astype("float32")
    # Frozen before inspecting this crop's class outcome: three genuine 3-D
    # oversegmentation regimes, with false splits preferred to false merges.
    regimes = ((0.30, 3), (0.40, 4), (0.50, 5))
    sv = [_supervoxels(boundary, cut, seed) for cut, seed in regimes]
    targets = np.zeros_like(affinity, dtype=np.uint8)
    mask = np.zeros_like(affinity, dtype=bool)
    counts = []
    for channel, offset in enumerate(OFFSETS_ZYX):
        left, right = _slices(raw.shape, offset)
        a, b = boundary[left], boundary[right]
        aff = affinity[(channel,) + left]
        same_sv = np.ones(aff.shape, dtype=bool)
        different_sv = np.ones(aff.shape, dtype=bool)
        for labels in sv:
            first, second = labels[left], labels[right]
            same_sv &= (first > 0) & (first == second)
            different_sv &= (first > 0) & (second > 0) & (first != second)
        # Same pairs must be inside all regimes, away from raw boundaries, and
        # already receive strong model continuity. Different pairs require a
        # persistent split plus raw boundary support and low model affinity.
        same = same_sv & (np.maximum(a, b) <= 0.20) & (aff >= 0.80)
        different = different_sv & (np.maximum(a, b) >= 0.70) & (aff <= 0.20)
        region_target, region_mask = targets[(channel,) + left], mask[(channel,) + left]
        region_target[same] = 1
        region_mask[same | different] = True
        counts.append({"channel": channel, "offset_zyx": list(offset), "same": int(same.sum()), "different": int(different.sum()), "ignore": int((~(same | different)).sum())})
    same_total = sum(item["same"] for item in counts)
    different_total = sum(item["different"] for item in counts)
    # Require both classes and enough spatially distributed boundary pairs.
    different_positions = np.argwhere((mask & (targets == 0)).any(axis=0))
    occupied_z = len(np.unique(different_positions[:, 0] // 8)) if len(different_positions) else 0
    admitted = same_total >= 128 and different_total >= 128 and occupied_z >= 3
    np.save(output_dir / "affinity_targets_czyx.npy", targets, allow_pickle=False)
    np.save(output_dir / "supervision_mask_czyx.npy", mask, allow_pickle=False)
    np.save(output_dir / "raw_boundary_zyx.npy", boundary, allow_pickle=False)
    for index, labels in enumerate(sv): np.save(output_dir / f"supervoxels_regime_{index}.npy", labels, allow_pickle=False)
    record = {"id": f"MV-AUTOPAIR-{crop_id[-6:]}", "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
              "status": "AUTO_PAIR_SUPERVISION_ADMITTED" if admitted else "AUTO_PAIR_SUPERVISION_REJECTED",
              "classification": "EXPERIMENTAL_PAIR_SUPERVISION_NOT_REVIEWED_OR_BIOLOGICAL", "crop_id": crop_id, "split": crop["split"],
              "raw_sha256": crop["raw_crop_sha256"], "affinity_sha256": sha256_file(affinity_path), "offsets_zyx": [list(item) for item in OFFSETS_ZYX],
              "supervoxel_regimes": [{"boundary_cut": cut, "seed_distance": seed, "true_3d": True, "instances": int(labels.max())} for (cut, seed), labels in zip(regimes, sv, strict=True)],
              "criteria": {"same": "same all supervoxels AND boundary<=0.20 AND affinity>=0.80", "different": "different all supervoxels AND boundary>=0.70 AND affinity<=0.20", "ignore": "all other pairs"},
              "counts": {"per_channel": counts, "same": same_total, "different": different_total, "ignore": sum(item["ignore"] for item in counts), "different_z_bins_of_8": occupied_z},
              "outputs": {"targets_sha256": sha256_file(output_dir / "affinity_targets_czyx.npy"), "mask_sha256": sha256_file(output_dir / "supervision_mask_czyx.npy")},
              "admission": {"admitted": admitted, "minimum_each_class": 128, "minimum_different_z_bins": 3},
              "prohibited_uses": ["reviewed_dvid_label", "biological_evidence", "mv_frag", "mv_neuron", "mv_conn"]}
    write_json_atomic(output_dir / "record.json", record)
    return record
