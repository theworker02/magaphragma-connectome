"""Acquire fresh C/E-scale crops for PRODUCTION_GT_BATCH_002 (predeclared sampling freeze).

Raw-only axis-balanced screen. Does not use Batch 001 labels or visual DIFFERENT targeting.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
OUT_DIR = MANIFEST.parent
FREEZE = REPO / "experiments/phase6e/PRODUCTION_GT_BATCH_002_SAMPLING_FREEZE.json"
RECEIPT = REPO / "experiments/phase6e/PRODUCTION_GT_BATCH_002_SOURCE_ACQUISITION.json"

BASE = "https://waspem-dvid2.flatironinstitute.org/api/node/aa49d16e88424cdbae89f4b8ced2a49b/five_yuri_4contrast/raw/0_1_2/"
DOMAIN_XYZ = (16648, 13544, 15401)
BLOCK_XYZ = (128, 128, 64)
N_TARGET = 8
N_FETCH_CAP = 24
MIN_GAP = 64
ABS_AXIS_MIN = 12.0
SEED = "PRODUCTION_GT_BATCH_002_SOURCE_ACQUISITION"
ANCHORS_XYZ = [
    (1984, 1600, 11456),
    (10304, 8384, 11456),
    (11328, 7552, 13440),
]


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def box_from_origin(o: tuple[int, int, int]) -> dict:
    x, y, z = o
    return {"x": [x, x + BLOCK_XYZ[0]], "y": [y, y + BLOCK_XYZ[1]], "z": [z, z + BLOCK_XYZ[2]]}


def separated(a: dict, b: dict, gap: int) -> bool:
    gaps = [max(b[ax][0] - a[ax][1], a[ax][0] - b[ax][1]) for ax in ("x", "y", "z")]
    return max(gaps) >= gap


def align64(v: int, limit: int, size: int) -> int:
    v = (v // 64) * 64
    return max(0, min(limit - size, v))


def candidate_origins(existing: list[dict]) -> list[tuple[int, int, int]]:
    rings = [128, 192, 256, 320, 384, 448, 512, 576, 640, 768, 896]
    dirs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if not (dx == dy == dz == 0)]
    raw = []
    for ax, ay, az in ANCHORS_XYZ:
        for r in rings:
            for dx, dy, dz in dirs:
                raw.append(
                    (
                        align64(ax + dx * r, DOMAIN_XYZ[0], BLOCK_XYZ[0]),
                        align64(ay + dy * r, DOMAIN_XYZ[1], BLOCK_XYZ[1]),
                        align64(az + dz * r, DOMAIN_XYZ[2], BLOCK_XYZ[2]),
                    )
                )
    uniq = sorted(set(raw), key=lambda o: hashlib.sha256(f"{SEED}|{o}".encode()).hexdigest())
    chosen = []
    boxes = list(existing)
    for o in uniq:
        box = box_from_origin(o)
        if any(not separated(box, e, MIN_GAP) for e in boxes):
            continue
        chosen.append(o)
        boxes.append(box)
        if len(chosen) >= N_FETCH_CAP:
            break
    return chosen


def axis_high_counts(arr: np.ndarray) -> dict[str, int | float]:
    m = 4
    z, y, x = arr.shape
    out: dict[str, int | float] = {}
    for ax, name in enumerate("ZYX"):
        if ax == 0:
            d = np.abs(arr[m + 1 : z - m, m : y - m, m : x - m] - arr[m : z - m - 1, m : y - m, m : x - m])
        elif ax == 1:
            d = np.abs(arr[m : z - m, m + 1 : y - m, m : x - m] - arr[m : z - m, m : y - m - 1, m : x - m])
        else:
            d = np.abs(arr[m : z - m, m : y - m, m + 1 : x - m] - arr[m : z - m, m : y - m, m : x - m - 1])
        out[name] = int((d >= ABS_AXIS_MIN).sum())
        out[f"{name}_max"] = float(d.max())
    return out


def main() -> int:
    if RECEIPT.exists():
        raise SystemExit(f"Receipt exists: {RECEIPT}")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_BATCH_002_LABELS":
        raise SystemExit("Batch 002 sampling freeze missing/invalid")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    existing = [r["bounds_xyz"] for r in manifest["records"]]
    next_ord = max(int(r["id"].rsplit("-", 1)[-1]) for r in manifest["records"]) + 1
    assert next_ord == 69, next_ord

    origins = candidate_origins(existing)
    if len(origins) < N_TARGET:
        raise SystemExit(f"Only {len(origins)} origins")

    expected = int(np.prod(BLOCK_XYZ))
    fetched = []
    for i, origin in enumerate(origins):
        ordinal = next_ord + i
        identifier = f"MV-DVID-RAW-SURVEY-{ordinal:03d}"
        url = BASE + f"{BLOCK_XYZ[0]}_{BLOCK_XYZ[1]}_{BLOCK_XYZ[2]}/{origin[0]}_{origin[1]}_{origin[2]}"
        print(f"fetching {identifier} {origin} ...", flush=True)
        with urlopen(url, timeout=180) as response:
            payload = response.read()
        if len(payload) != expected:
            raise RuntimeError(f"{identifier}: bad size")
        raw = np.frombuffer(payload, dtype=np.uint8).reshape((BLOCK_XYZ[2], BLOCK_XYZ[1], BLOCK_XYZ[0]))
        path = OUT_DIR / f"{identifier}-raw_zyx.npy"
        if path.exists():
            raise FileExistsError(path)
        np.save(path, raw, allow_pickle=False)
        low, high = np.quantile(raw, (0.005, 0.995))
        counts = axis_high_counts(raw.astype(np.float64))
        balanced = all(int(counts[a]) >= 8 for a in ("Z", "Y", "X"))
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
                "batch_id": "PRODUCTION_GT_BATCH_002_SOURCE_ACQUISITION",
                "purpose": "CANDIDATE_PRODUCTION_GT_BATCH_002",
                "axis_abs_ge12_counts": counts,
                "axis_balanced_ce_scale": balanced,
            },
        }
        fetched.append(rec)
        existing.append(rec["bounds_xyz"])

    balanced_only = [r for r in fetched if r["acquisition"]["axis_balanced_ce_scale"]]
    pool = balanced_only if len(balanced_only) >= N_TARGET else fetched
    ranked = sorted(
        pool,
        key=lambda r: (
            0 if r["acquisition"]["axis_balanced_ce_scale"] else 1,
            -min(int(r["acquisition"]["axis_abs_ge12_counts"][a]) for a in ("Z", "Y", "X")),
            -r["intensity"]["std"],
        ),
    )
    keep = ranked[:N_TARGET]
    keep_ids = {r["id"] for r in keep}

    manifest["records"].extend(fetched)
    manifest["extended_at"] = _now()
    manifest["extension_pgt002"] = {
        "id": "PRODUCTION_GT_BATCH_002_SOURCE_ACQUISITION",
        "n_added": len(fetched),
        "ids": [r["id"] for r in fetched],
        "selected_for_batch_002": sorted(keep_ids),
    }
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(MANIFEST)

    receipt = {
        "id": "PRODUCTION_GT_BATCH_002_SOURCE_ACQUISITION",
        "created_at": _now(),
        "status": "ACQUIRED",
        "sampling_freeze": str(FREEZE.relative_to(REPO)).replace("\\", "/"),
        "n_fetched": len(fetched),
        "n_selected": len(keep),
        "n_axis_balanced": len(balanced_only),
        "selected_ids": sorted(keep_ids),
        "reserve_ids": sorted(r["id"] for r in fetched if r["id"] not in keep_ids),
        "per_selected": [
            {
                "id": r["id"],
                "raw_sha256": r["raw_sha256"],
                "bounds_xyz": r["bounds_xyz"],
                "axis_abs_ge12_counts": r["acquisition"]["axis_abs_ge12_counts"],
                "axis_balanced_ce_scale": r["acquisition"]["axis_balanced_ce_scale"],
                "intensity_std": r["intensity"]["std"],
            }
            for r in keep
        ],
        "prohibited_inputs": ["batch_001_labels", "human_labels", "visual_different_targeting", "model_predictions"],
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"fetched": len(fetched), "balanced": len(balanced_only), "selected": sorted(keep_ids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
