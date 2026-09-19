"""Conservative classical membrane-boundary / watershed baseline for real EM crops."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .io import sha256_file, write_json_atomic


MODEL_ID = "MV-SEG-MODEL-0001"


def run_watershed_baseline(ingestion_path: Path, output_root: Path) -> Path:
    ingestion = json.loads(ingestion_path.read_text(encoding="utf-8"))
    raw = np.load(ingestion["raw_array"], mmap_mode="r")
    # Bounded inner crop avoids edge artifacts while retaining source coordinate registration.
    z, y, x = raw.shape
    crop = np.asarray(raw[max(0, z//2-16):min(z, z//2+16), y//2-256:y//2+256, x//2-256:x//2+256], dtype=np.float32)
    started = time.time()
    lower, upper = np.percentile(crop, (1, 99))
    normalized = np.clip((crop-lower) / max(upper-lower, 1e-6), 0, 1)
    smoothed = ndimage.gaussian_filter(normalized, sigma=(0.8, 1.0, 1.0))
    gradients = np.gradient(smoothed)
    membrane = np.sqrt(sum(component * component for component in gradients))
    membrane_u8 = np.clip(membrane / max(float(membrane.max()), 1e-8) * 255, 0, 255).astype(np.uint8)
    # Marker-controlled watershed over a boundary-energy volume, an established baseline family.
    seed_grid = np.zeros(crop.shape, dtype=np.int32)
    serial = 1
    for zz in range(2, crop.shape[0], 8):
        for yy in range(16, crop.shape[1], 64):
            for xx in range(16, crop.shape[2], 64):
                if membrane_u8[zz, yy, xx] < np.percentile(membrane_u8, 65):
                    seed_grid[zz, yy, xx] = serial
                    serial += 1
    labels = ndimage.watershed_ift(membrane_u8, seed_grid)
    result_dir = output_root / ingestion["volume_id"] / ingestion["region_id"] / "run-MV-SEG-RUN-000001"
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / "membrane_probability_u8.npy", membrane_u8)
    np.save(result_dir / "labels_zyx.npy", labels)
    middle = crop.shape[0] // 2
    Image.fromarray(crop[middle].astype(np.uint8)).save(result_dir / "raw_xy_z6032.png")
    Image.fromarray(membrane_u8[middle]).save(result_dir / "membrane_xy_z6032.png")
    colors = np.stack(((labels[middle] * 37) % 255, (labels[middle] * 97) % 255, (labels[middle] * 173) % 255), axis=-1).astype(np.uint8)
    overlay = (0.55 * np.repeat(crop[middle, :, :, None], 3, axis=2) + 0.45 * colors).astype(np.uint8)
    Image.fromarray(overlay).save(result_dir / "overlay_xy_z6032.png")
    origin = ingestion["source_bounds_xyz"]
    crop_origin = [origin["x"][0] + x//2-256, origin["y"][0] + y//2-256, origin["z"][0] + max(0, z//2-16)]
    segments = []
    for label in np.unique(labels):
        if label <= 0:
            continue
        positions = np.argwhere(labels == label)
        if len(positions) < 128:
            continue
        lo, hi = positions.min(axis=0), positions.max(axis=0) + 1
        centroid_zyx = positions.mean(axis=0)
        segment_id = f"MV-SEG-{len(segments)+1:08d}"
        segments.append({
            "id": segment_id, "evidence_ids": ["MV-EV-000002"], "dataset_ids": [ingestion["dataset_id"]],
            "status": "MACHINE_SEGMENTED", "review_state": "MACHINE_ONLY", "volume_id": ingestion["volume_id"],
            "coordinates_nm_xyz": [(crop_origin[0]+float(centroid_zyx[2]))*8, (crop_origin[1]+float(centroid_zyx[1]))*8, (crop_origin[2]+float(centroid_zyx[0]))*8],
            "voxel_bounds_xyz": {"x":[crop_origin[0]+int(lo[2]), crop_origin[0]+int(hi[2])], "y":[crop_origin[1]+int(lo[1]), crop_origin[1]+int(hi[1])], "z":[crop_origin[2]+int(lo[0]), crop_origin[2]+int(hi[0])]},
            "voxel_count": int(len(positions)), "model_id": MODEL_ID, "run_id": "MV-SEG-RUN-000001", "confidence": None,
        })
    write_json_atomic(result_dir / "segments.json", {"segments": segments})
    run = {"id":"MV-SEG-RUN-000001", "model_id":MODEL_ID, "method":"gradient-derived membrane proxy + marker-controlled watershed", "input_sha256":ingestion["raw_sha256"], "outputs":{"labels_sha256":sha256_file(result_dir / "labels_zyx.npy"), "membrane_sha256":sha256_file(result_dir / "membrane_probability_u8.npy")}, "crop_origin_xyz":crop_origin, "crop_shape_zyx":list(crop.shape), "runtime_seconds":time.time()-started, "hardware":{"backend":"CPU", "platform":platform.platform()}, "review_state":"MACHINE_ONLY"}
    write_json_atomic(result_dir / "run.json", run)
    return result_dir
