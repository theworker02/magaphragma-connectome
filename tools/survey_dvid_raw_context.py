"""Download a bounded, stratified raw-DVID visual-QA survey.

This is a source-content survey, not a segmentation or annotation generator.
It never uses model output, produces no labels, and records every sampled raw
block plus its source bounds and content hash for later human/expert review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw


BASE = "https://waspem-dvid2.flatironinstitute.org/api/node/aa49d16e88424cdbae89f4b8ced2a49b/five_yuri_4contrast/raw/0_1_2/"
DOMAIN_XYZ = (16648, 13544, 15401)
BLOCK_XYZ = (128, 128, 64)
GRID = (4, 4, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def aligned_start(index: int, cells: int, limit: int, size: int) -> int:
    center = round((index + 0.5) * limit / cells - size / 2)
    return max(0, min(limit - size, (center // 64) * 64))


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a local-only raw-DVID visual survey")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--origins-json",
        type=Path,
        help="Optional JSON list of explicit native XYZ origins for a bounded raw-content screen.",
    )
    parser.add_argument(
        "--block-xyz",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=BLOCK_XYZ,
        help="Block shape used with --origins-json (default: %(default)s).",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite survey output: {args.output}")
    args.output.mkdir(parents=True)
    block_xyz = tuple(args.block_xyz)
    if args.origins_json:
        decoded = json.loads(args.origins_json.read_text(encoding="utf-8"))
        if not isinstance(decoded, list) or not decoded:
            raise SystemExit("--origins-json must contain a non-empty JSON list of [X,Y,Z] origins")
        starts = [tuple(int(value) for value in origin) for origin in decoded]
        if any(len(start) != 3 for start in starts):
            raise SystemExit("Every explicit origin must contain exactly three coordinates")
        if any(any(value < 0 or value + size > limit for value, size, limit in zip(start, block_xyz, DOMAIN_XYZ, strict=True)) for start in starts):
            raise SystemExit("An explicit origin falls outside the DVID volume")
        selection_method = "LOW_RESOLUTION_RAW_EM_CONTEXT_SCREENING_ONLY"
    else:
        starts = []
        for iz in range(GRID[2]):
            for iy in range(GRID[1]):
                for ix in range(GRID[0]):
                    starts.append(tuple(aligned_start(i, n, limit, size) for i, n, limit, size in zip((ix, iy, iz), GRID, DOMAIN_XYZ, BLOCK_XYZ, strict=True)))
        selection_method = "FIXED_SPATIALLY_STRATIFIED_GRID_V1"
    records = []
    thumbnails: list[tuple[str, Image.Image]] = []
    expected = int(np.prod(block_xyz))
    for ordinal, start in enumerate(starts, 1):
        identifier = f"MV-DVID-RAW-SURVEY-{ordinal:03d}"
        url = BASE + f"{block_xyz[0]}_{block_xyz[1]}_{block_xyz[2]}/{start[0]}_{start[1]}_{start[2]}"
        with urlopen(url, timeout=180) as response:
            payload = response.read()
        if len(payload) != expected:
            raise RuntimeError(f"{identifier}: response {len(payload)} bytes, expected {expected}")
        raw = np.frombuffer(payload, dtype=np.uint8).reshape((block_xyz[2], block_xyz[1], block_xyz[0]))
        path = args.output / f"{identifier}-raw_zyx.npy"
        np.save(path, raw, allow_pickle=False)
        low, high = np.quantile(raw, (0.005, 0.995))
        image = np.clip((raw[raw.shape[0] // 2].astype(np.float32) - low) * 255 / max(high - low, 1), 0, 255).astype(np.uint8)
        thumbnails.append((identifier, Image.fromarray(image).resize((192, 192))))
        records.append({
            "id": identifier, "status": "RAW_DVID_VISUAL_QA_ONLY", "source_url": url,
            "bounds_xyz": {"x": [start[0], start[0] + block_xyz[0]], "y": [start[1], start[1] + block_xyz[1]], "z": [start[2], start[2] + block_xyz[2]]},
            "shape_zyx": list(raw.shape), "raw_path": str(path.resolve()), "raw_sha256": sha256(path),
            "dtype": str(raw.dtype), "intensity": {"min": int(raw.min()), "max": int(raw.max()), "mean": float(raw.mean()), "std": float(raw.std()), "p005": float(low), "p995": float(high)},
            "prohibited_promotions": ["SAME_PROCESS", "DIFFERENT_PROCESS", "REVIEWED_DVID_LABEL", "MV-FRAG", "MV-N", "MV-SYN", "MV-CONN"],
        })
    atlas = Image.new("L", (192 * 4, 192 * 8)); draw = ImageDraw.Draw(atlas)
    for index, (identifier, image) in enumerate(thumbnails):
        x, y = index % 4 * 192, index // 4 * 192
        atlas.paste(image, (x, y)); draw.text((x + 3, y + 3), identifier[-3:], fill=255)
    atlas_path = args.output / "survey-mid-z-atlas-windowed.png"; atlas.save(atlas_path)
    manifest = {"id": "MV-DVID-RAW-CONTEXT-SURVEY-001", "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "status": "RAW_DVID_VISUAL_QA_REQUIRED", "source": {"instance": "five_yuri_4contrast", "voxel_size_nm_xyz": [8, 8, 8]},
                "selection": {"method": selection_method, "grid_xyz": list(GRID) if not args.origins_json else None, "block_xyz": list(block_xyz), "prohibited_inputs": ["SegNeuron", "pretrained_models", "image_feature_ranking", "prior_review_decisions"]},
                "records": records, "atlas_path": str(atlas_path.resolve()), "atlas_sha256": sha256(atlas_path),
                "local_cache_only": True, "redistribution_status": "UNRESOLVED_DO_NOT_EXPORT"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"blocks": len(records), "atlas": str(atlas_path), "manifest": str((args.output / 'manifest.json').resolve())}))


if __name__ == "__main__":
    main()
