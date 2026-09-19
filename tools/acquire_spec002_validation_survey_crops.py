"""Acquire ≥12 new never-reviewed DVID raw survey crops for SPEC002 validation.

Appends MV-DVID-RAW-SURVEY-033+ into the existing survey_001 cache/manifest.
Selection is label-blind: spatial independence from all existing survey boxes
and known G3 region boxes only. No model/label inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
OUT_DIR = MANIFEST.parent
RECEIPT = REPO / "experiments/phase6e/AFFINITY_SPEC002_VALIDATION_SOURCE_ACQUISITION_001.json"

BASE = "https://waspem-dvid2.flatironinstitute.org/api/node/aa49d16e88424cdbae89f4b8ced2a49b/five_yuri_4contrast/raw/0_1_2/"
DOMAIN_XYZ = (16648, 13544, 15401)
BLOCK_XYZ = (128, 128, 64)
N_NEW = 16
MIN_GAP = 64  # voxels L-inf style margin between boxes
SEED = "AFFINITY_SPEC002_VALIDATION_SOURCE_ACQUISITION_001"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def box_from_origin(origin_xyz: tuple[int, int, int]) -> dict:
    x, y, z = origin_xyz
    return {
        "x": [x, x + BLOCK_XYZ[0]],
        "y": [y, y + BLOCK_XYZ[1]],
        "z": [z, z + BLOCK_XYZ[2]],
    }


def separated(a: dict, b: dict, gap: int) -> bool:
    # True if boxes are separated by >= gap along at least one axis (L-inf exterior gap).
    axis_gaps = []
    for ax in ("x", "y", "z"):
        axis_gaps.append(max(b[ax][0] - a[ax][1], a[ax][0] - b[ax][1]))
    # Separated if the maximum of the (possibly negative) axis gaps is >= gap
    # For non-overlap we need all pairwise: actually for AABB separation in 3D,
    # boxes are non-overlapping if they separate on ANY axis. With margin:
    return max(axis_gaps) >= gap


def all_existing_boxes(manifest: dict) -> list[dict]:
    boxes = []
    for rec in manifest["records"]:
        boxes.append({"id": rec["id"], "bounds_xyz": rec["bounds_xyz"]})
    # Also include workspace parent bounds already in survey records — sufficient.
    return boxes


def candidate_origins() -> list[tuple[int, int, int]]:
    """Dense stratified candidates, deterministically ordered by hash."""
    # 8 x 8 x 4 grid centers → 256 candidates
    grid = (8, 8, 4)
    starts = []
    for iz in range(grid[2]):
        for iy in range(grid[1]):
            for ix in range(grid[0]):
                origin = []
                for i, n, limit, size in zip((ix, iy, iz), grid, DOMAIN_XYZ, BLOCK_XYZ, strict=True):
                    center = round((i + 0.5) * limit / n - size / 2)
                    origin.append(max(0, min(limit - size, (center // 64) * 64)))
                starts.append(tuple(origin))
    starts.sort(key=lambda o: hashlib.sha256(f"{SEED}|{o[0]},{o[1]},{o[2]}".encode()).hexdigest())
    return starts


def main() -> int:
    if RECEIPT.exists():
        raise SystemExit(f"Acquisition receipt already exists: {RECEIPT}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    existing_ids = {r["id"] for r in manifest["records"]}
    if any(f"MV-DVID-RAW-SURVEY-{i:03d}" in existing_ids for i in range(33, 33 + N_NEW)):
        raise SystemExit("Target SURVEY-033+ ids already present in manifest")

    existing = all_existing_boxes(manifest)
    chosen: list[tuple[int, int, int]] = []
    chosen_boxes: list[dict] = []
    for origin in candidate_origins():
        box = box_from_origin(origin)
        if any(not separated(box, e["bounds_xyz"], MIN_GAP) for e in existing):
            continue
        if any(not separated(box, cb, MIN_GAP) for cb in chosen_boxes):
            continue
        # also stay inside domain (already ensured by construction)
        chosen.append(origin)
        chosen_boxes.append(box)
        if len(chosen) >= N_NEW:
            break
    if len(chosen) < N_NEW:
        raise SystemExit(f"Only found {len(chosen)} non-overlapping origins; need {N_NEW}")

    next_ordinal = max(int(r["id"].rsplit("-", 1)[-1]) for r in manifest["records"]) + 1
    assert next_ordinal == 33, next_ordinal

    new_records = []
    expected = int(np.prod(BLOCK_XYZ))
    for i, origin in enumerate(chosen):
        ordinal = next_ordinal + i
        identifier = f"MV-DVID-RAW-SURVEY-{ordinal:03d}"
        url = BASE + f"{BLOCK_XYZ[0]}_{BLOCK_XYZ[1]}_{BLOCK_XYZ[2]}/{origin[0]}_{origin[1]}_{origin[2]}"
        print(f"fetching {identifier} {origin} ...", flush=True)
        with urlopen(url, timeout=180) as response:
            payload = response.read()
        if len(payload) != expected:
            raise RuntimeError(f"{identifier}: got {len(payload)} bytes, expected {expected}")
        raw = np.frombuffer(payload, dtype=np.uint8).reshape((BLOCK_XYZ[2], BLOCK_XYZ[1], BLOCK_XYZ[0]))
        path = OUT_DIR / f"{identifier}-raw_zyx.npy"
        if path.exists():
            raise FileExistsError(path)
        np.save(path, raw, allow_pickle=False)
        low, high = np.quantile(raw, (0.005, 0.995))
        rec = {
            "id": identifier,
            "status": "RAW_DVID_VISUAL_QA_ONLY",
            "source_url": url,
            "bounds_xyz": box_from_origin(origin),
            "shape_zyx": list(raw.shape),
            "raw_path": str(path.resolve()),
            "raw_sha256": sha256(path),
            "dtype": str(raw.dtype),
            "intensity": {
                "min": int(raw.min()),
                "max": int(raw.max()),
                "mean": float(raw.mean()),
                "std": float(raw.std()),
                "p005": float(low),
                "p995": float(high),
            },
            "prohibited_promotions": [
                "SAME_PROCESS",
                "DIFFERENT_PROCESS",
                "REVIEWED_DVID_LABEL",
                "MV-FRAG",
                "MV-N",
                "MV-SYN",
                "MV-CONN",
            ],
            "acquisition": {
                "batch_id": "AFFINITY_SPEC002_VALIDATION_SOURCE_ACQUISITION_001",
                "purpose": "SPEC002_ANNOTATION_PROCEDURE_VALIDATION_ONLY",
                "min_gap_voxels": MIN_GAP,
            },
        }
        new_records.append(rec)
        existing.append({"id": identifier, "bounds_xyz": rec["bounds_xyz"]})

    manifest["records"].extend(new_records)
    manifest["extended_at"] = _now()
    manifest["extension"] = {
        "id": "AFFINITY_SPEC002_VALIDATION_SOURCE_ACQUISITION_001",
        "n_added": len(new_records),
        "ids": [r["id"] for r in new_records],
    }
    # atomic-ish write
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(MANIFEST)

    receipt = {
        "id": "AFFINITY_SPEC002_VALIDATION_SOURCE_ACQUISITION_001",
        "created_at": _now(),
        "status": "ACQUIRED",
        "n_requested": N_NEW,
        "n_acquired": len(new_records),
        "min_gap_voxels": MIN_GAP,
        "block_xyz": list(BLOCK_XYZ),
        "seed": SEED,
        "selection_method": "HASH_ORDERED_DENSE_GRID_NONOVERLAP_V1",
        "prohibited_inputs": [
            "human_labels",
            "g3_labels",
            "smoke_labels",
            "model_predictions",
            "segmentation",
        ],
        "records": [
            {
                "id": r["id"],
                "raw_sha256": r["raw_sha256"],
                "bounds_xyz": r["bounds_xyz"],
                "source_url": r["source_url"],
            }
            for r in new_records
        ],
        "manifest_path": str(MANIFEST.relative_to(REPO)).replace("\\", "/"),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"acquired": len(new_records), "ids": [r["id"] for r in new_records], "receipt": str(RECEIPT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
