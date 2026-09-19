"""Open ROI after S4 authorization: materialize crop, dense aff, true-3D seg, machine QC.

Requires AFFINITY_S4_ROI_AUTHORIZATION_001 APPROVED. Does not open full-volume (S7).
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from urllib.request import urlopen

import numpy as np

REPO = Path(__file__).resolve().parents[1]

S4 = REPO / "experiments/phase6e/AFFINITY_S4_ROI_AUTHORIZATION_001.json"
ROI_GATE = REPO / "experiments/phase6e/AFFINITY_ROI_RECONSTRUCTION_GATE_001.json"
QC_CONTRACT = REPO / "experiments/phase6e/SEG_ROI_QC_CONTRACT_001.json"
FUNNEL_SUMMARY = REPO / "experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/FUNNEL_SUMMARY.json"
MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
OUT = REPO / "experiments/phase6e/AFFINITY-ROI-001"
CKPT = REPO / "experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt"

BASE = "https://waspem-dvid2.flatironinstitute.org/api/node/aa49d16e88424cdbae89f4b8ced2a49b/five_yuri_4contrast/raw/0_1_2/"
DOMAIN_XYZ = (16648, 13544, 15401)
ROI_XYZ = (256, 256, 64)  # X,Y,Z → stored as ZYX (64,256,256)
SEED = "AFFINITY_ROI_RECONSTRUCTION_GATE_001|ROI"
TRAIN_SOURCES = [
    "MV-DVID-RAW-SURVEY-076",
    "MV-DVID-RAW-SURVEY-079",
    "MV-DVID-RAW-SURVEY-082",
    "MV-DVID-RAW-SURVEY-087",
    "MV-DVID-RAW-SURVEY-088",
    "MV-DVID-RAW-SURVEY-092",
]
EVAL_BLOCKED = {"MV-DVID-RAW-SURVEY-078", "MV-DVID-RAW-SURVEY-083"}
CROP_SIZE = (20, 64, 64)
STRIDE = (10, 32, 32)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def win_to_wsl(path: Path | str) -> Path:
    text = str(path).replace("\\", "/")
    on_linux = sys.platform.startswith("linux")
    if text.startswith("/mnt/") and not on_linux:
        parts = text.split("/")
        drive = parts[2].upper()
        rest = "/".join(parts[3:])
        return Path(f"{drive}:/{rest}")
    if len(text) >= 2 and text[1] == ":" and on_linux:
        return Path(f"/mnt/{text[0].lower()}/{text[3:]}")
    return Path(text)


def align64(v: int, limit: int, size: int) -> int:
    v = (v // 64) * 64
    return max(0, min(limit - size, v))


def select_source(manifest: dict) -> dict:
    by_id = {r["id"]: r for r in manifest["records"]}
    eligible = []
    for sid in TRAIN_SOURCES:
        if sid in EVAL_BLOCKED:
            raise SystemExit(f"eval source leaked into ROI pool: {sid}")
        if sid not in by_id:
            raise SystemExit(f"missing survey record {sid}")
        eligible.append(by_id[sid])
    ranked = sorted(eligible, key=lambda r: hashlib.sha256(f"{SEED}|{r['id']}".encode()).hexdigest())
    return ranked[0]


def roi_origin_xyz(bounds: dict) -> tuple[int, int, int]:
    cx = (bounds["x"][0] + bounds["x"][1]) // 2
    cy = (bounds["y"][0] + bounds["y"][1]) // 2
    cz = (bounds["z"][0] + bounds["z"][1]) // 2
    return (
        align64(cx - ROI_XYZ[0] // 2, DOMAIN_XYZ[0], ROI_XYZ[0]),
        align64(cy - ROI_XYZ[1] // 2, DOMAIN_XYZ[1], ROI_XYZ[1]),
        align64(cz - ROI_XYZ[2] // 2, DOMAIN_XYZ[2], ROI_XYZ[2]),
    )


def fetch_roi(origin: tuple[int, int, int]) -> tuple[np.ndarray, str]:
    url = BASE + f"{ROI_XYZ[0]}_{ROI_XYZ[1]}_{ROI_XYZ[2]}/{origin[0]}_{origin[1]}_{origin[2]}"
    expected = int(np.prod(ROI_XYZ))
    with urlopen(url, timeout=300) as response:
        payload = response.read()
    if len(payload) != expected:
        raise RuntimeError(f"bad DVID payload size {len(payload)} != {expected}")
    raw = np.frombuffer(payload, dtype=np.uint8).reshape((ROI_XYZ[2], ROI_XYZ[1], ROI_XYZ[0]))
    return raw, url


def tile_layout(shape):
    counts = tuple(max(1, (n - c) // s + 2) for n, c, s in zip(shape, CROP_SIZE, STRIDE))
    padded_shape = tuple(c + (count - 1) * s for c, count, s in zip(CROP_SIZE, counts, STRIDE))
    padding = tuple(((p - n) // 2, (p - n + 1) // 2) for n, p in zip(shape, padded_shape))
    starts = tuple(tuple(i * step for i in range(count)) for count, step in zip(counts, STRIDE))
    return padding, starts


def gaussian_weight():
    zz, yy, xx = np.meshgrid(*(np.linspace(-1, 1, n, dtype=np.float32) for n in CROP_SIZE), indexing="ij")
    distance = np.sqrt(zz * zz + yy * yy + xx * xx)
    return 1e-6 + np.exp(-(distance**2 / (2.0 * 0.2**2)))


def infer_volume(model, volume: np.ndarray, device: str):
    import torch

    padding, starts = tile_layout(volume.shape)
    padded = np.pad(volume, padding, mode="reflect")
    sums = np.zeros((4,) + padded.shape, dtype=np.float32)
    weights = np.zeros(padded.shape, dtype=np.float32)
    patch_weight = gaussian_weight()
    total = len(starts[0]) * len(starts[1]) * len(starts[2])
    model = model.to(device).eval()
    with torch.inference_mode():
        for index, position in enumerate(product(*starts), 1):
            region = tuple(slice(start, start + size) for start, size in zip(position, CROP_SIZE))
            patch = np.ascontiguousarray(padded[region], dtype=np.float32) / 255.0
            tensor = torch.from_numpy(patch[None, None]).to(device)
            affinities, boundaries = model(tensor)
            prediction = torch.cat((affinities, boundaries), dim=1)[0].float().cpu().numpy()
            sums[(slice(None),) + region] += prediction * patch_weight
            weights[region] += patch_weight
            if index % 50 == 0 or index == total:
                print(f"  infer {index}/{total}", flush=True)
    if not np.all(weights > 0):
        raise RuntimeError("Inference grid did not cover the volume")
    sums /= weights[None]
    np.clip(sums, 0.0, 1.0, out=sums)
    original = tuple(slice(pad[0], pad[0] + size) for pad, size in zip(padding, volume.shape))
    return sums[(slice(0, 3),) + original].copy(), sums[(3,) + original].copy()


def run_true3d_seg(affinities: np.ndarray, boundaries: np.ndarray, out_dir: Path, beta: float = 0.1) -> dict:
    import elf.segmentation.features as features
    import elf.segmentation.multicut as multicut
    import elf.segmentation.watershed as watershed

    fused = np.minimum(affinities, boundaries[None, ...])
    split_affinities = 1.0 - fused
    watershed_input = np.maximum(split_affinities[1], split_affinities[2])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fragments, _ = watershed.distance_transform_watershed(
            watershed_input, threshold=0.25, sigma_seeds=2.0, pixel_pitch=(1.0, 1.0, 1.0)
        )
    fragments = np.asarray(fragments, dtype=np.uint32)
    rag = features.compute_rag(fragments)
    if rag.numberOfEdges == 0:
        node_labels = np.arange(rag.numberOfNodes, dtype=np.uint64)
        affinity_feature = np.zeros(0, dtype=np.float64)
        edge_sizes = np.zeros(0, dtype=np.float64)
        costs = np.zeros(0, dtype=np.float64)
    else:
        affinity_feature = features.compute_affinity_features(
            rag,
            fragments,
            split_affinities,
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        )[:, 0]
        edge_sizes = features.compute_boundary_mean_and_length(rag, fragments, watershed_input)[:, 1]
        costs = multicut.transform_probabilities_to_costs(affinity_feature, edge_sizes=edge_sizes, beta=beta)
        node_labels = multicut.multicut_kernighan_lin(rag, costs)
    labels = features.project_node_labels_to_pixels(rag, fragments, node_labels)
    _, inverse = np.unique(labels, return_inverse=True)
    labels = (inverse.reshape(fragments.shape) + 1).astype(np.uint32)
    np.save(out_dir / "fragments.npy", fragments, allow_pickle=False)
    np.save(out_dir / "labels.npy", labels, allow_pickle=False)
    return {
        "watershed": {
            "algorithm": "ELF distance_transform_watershed",
            "dimensions": "3D",
            "threshold": 0.25,
            "sigma_seeds": 2.0,
            "warnings": [str(item.message) for item in caught],
        },
        "fragments": {"count": int(np.unique(fragments).size)},
        "rag": {"nodes": int(rag.numberOfNodes), "edges": int(rag.numberOfEdges)},
        "instances": {"count": int(np.unique(labels).size)},
        "beta": beta,
    }


def z_extent(labels: np.ndarray) -> np.ndarray:
    result = []
    for label in np.unique(labels):
        if label == 0:
            continue
        positions = np.where(labels == label)[0]
        result.append(int(positions.max() - positions.min() + 1))
    return np.asarray(result, dtype=np.float64) if result else np.asarray([], dtype=np.float64)


def machine_qc(affinities: np.ndarray, labels: np.ndarray, seg_meta: dict, shape_zyx: list[int]) -> dict:
    qc_a = (
        affinities.shape == (3, *shape_zyx)
        and bool(np.isfinite(affinities).all())
        and float(affinities.min()) >= 0
        and float(affinities.max()) <= 1
    )
    uniq = [int(u) for u in np.unique(labels) if int(u) != 0]
    qc_b = len(uniq) >= 2
    sizes = np.asarray([(labels == u).sum() for u in uniq], dtype=np.int64) if uniq else np.asarray([], dtype=np.int64)
    tiny_frac = float((sizes < 512).mean()) if sizes.size else 1.0
    qc_c = tiny_frac < 0.95
    ze = z_extent(labels)
    qc_d = bool(ze.size) and float(np.median(ze)) >= 2.0
    qc_e = int(seg_meta["rag"]["edges"]) >= 1 or int(seg_meta["fragments"]["count"]) >= 2
    gates = {
        "QC_A_affinity_completeness": qc_a,
        "QC_B_not_single_instance": qc_b,
        "QC_C_not_all_tiny": qc_c,
        "QC_D_z_extent_nonzero": qc_d,
        "QC_E_fragment_rag_nonempty": qc_e,
    }
    return {
        "gates": gates,
        "metrics": {
            "n_instances": len(uniq),
            "tiny_instance_fraction_lt_512": tiny_frac,
            "median_z_extent": float(np.median(ze)) if ze.size else None,
            "fragment_count": seg_meta["fragments"]["count"],
            "rag_edges": seg_meta["rag"]["edges"],
        },
        "verdict": "MACHINE_QC_PASS" if all(gates.values()) else "MACHINE_QC_FAIL",
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "materialize_infer", "segment_qc"], default="all")
    args = ap.parse_args()

    s4 = json.loads(S4.read_text(encoding="utf-8"))
    if s4.get("status") != "APPROVED" or s4.get("human_decision") != "AUTHORIZE_ROI_OPEN":
        raise SystemExit("S4 authorization missing")
    if not CKPT.exists():
        raise SystemExit(f"missing checkpoint {CKPT}")

    if args.phase in ("all", "materialize_infer"):
        if OUT.exists():
            raise FileExistsError(OUT)
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA/ROCm required for ROI affinity inference")

        OUT.mkdir(parents=True)
        package = OUT / "package"
        package.mkdir()
        infer_dir = OUT / "inference"
        infer_dir.mkdir()

        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source = select_source(manifest)
        origin = roi_origin_xyz(source["bounds_xyz"])
        print(f"Selected source {source['id']} -> ROI origin_xyz={origin}", flush=True)
        raw, url = fetch_roi(origin)
        raw_path = package / "roi_raw_zyx.npy"
        np.save(raw_path, raw, allow_pickle=False)
        raw_sha = sha256(raw_path)

        package_meta = {
            "id": "AFFINITY_ROI_001_PACKAGE",
            "created_at": _now(),
            "s4_authorization": s4["id"],
            "anchor_survey_source_id": source["id"],
            "anchor_survey_raw_sha256": source["raw_sha256"],
            "anchor_bounds_xyz": source["bounds_xyz"],
            "roi_origin_xyz": list(origin),
            "roi_size_xyz": list(ROI_XYZ),
            "shape_zyx": list(raw.shape),
            "source_url": url,
            "raw_path": str(raw_path.resolve()),
            "raw_sha256": raw_sha,
            "eval_sources_excluded": sorted(EVAL_BLOCKED),
            "seed_material": SEED,
            "authorized_checkpoint_sha256": s4["authorized_candidate"]["selected_checkpoint_sha256"],
        }
        (package / "ROI_PACKAGE.json").write_text(json.dumps(package_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        gate = json.loads(ROI_GATE.read_text(encoding="utf-8"))
        gate["status"] = "OPEN_FOR_S5"
        gate["opened_by"] = s4["id"]
        gate["opened_at"] = _now()
        gate["materialized_package"] = str((package / "ROI_PACKAGE.json").relative_to(REPO)).replace("\\", "/")
        gate["explicit_origin_zyx_recorded"] = {
            "anchor_source_id": source["id"],
            "origin_xyz": list(origin),
            "shape_zyx": list(raw.shape),
            "raw_sha256": raw_sha,
        }
        ROI_GATE.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print("Dense affinity inference...", flush=True)
        sys.path.insert(0, str(REPO / "third_party/segneuron/Train_and_Inference"))
        from model.Mnet import MNet

        model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub")
        state = torch.load(win_to_wsl(CKPT), map_location="cuda", weights_only=False)
        model.load_state_dict(state["model_weights"], strict=True)
        affinities, boundaries = infer_volume(model, raw, device="cuda")
        aff_path = infer_dir / "affinities.npy"
        bnd_path = infer_dir / "boundaries.tif"
        np.save(aff_path, affinities, allow_pickle=False)
        import tifffile

        tifffile.imwrite(bnd_path, boundaries)
        inference_receipt = {
            "created_at": _now(),
            "checkpoint": str(CKPT).replace("\\", "/"),
            "checkpoint_sha256": sha256(win_to_wsl(CKPT)),
            "crop_size_zyx": list(CROP_SIZE),
            "stride_zyx": list(STRIDE),
            "affinities_shape_czyx": list(affinities.shape),
            "boundaries_shape_zyx": list(boundaries.shape),
            "affinities_sha256": sha256(aff_path),
            "raw_sha256": raw_sha,
        }
        (infer_dir / "inference.json").write_text(
            json.dumps(inference_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (OUT / "PHASE_MATERIALIZE_INFER_DONE.json").write_text(
            json.dumps({"created_at": _now(), "package": package_meta, "inference": inference_receipt}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"phase": "materialize_infer", "source": source["id"], "origin_xyz": list(origin)}, indent=2))
        if args.phase == "materialize_infer":
            return 0

    if args.phase in ("all", "segment_qc"):
        if not OUT.exists():
            raise SystemExit("run materialize_infer first")
        package_meta = json.loads((OUT / "package" / "ROI_PACKAGE.json").read_text(encoding="utf-8"))
        inference_receipt = json.loads((OUT / "inference" / "inference.json").read_text(encoding="utf-8"))
        aff_path = OUT / "inference" / "affinities.npy"
        bnd_path = OUT / "inference" / "boundaries.tif"
        seg_dir = OUT / "segmentation"
        if seg_dir.exists():
            raise FileExistsError(seg_dir)
        seg_dir.mkdir()

        affinities = np.load(aff_path, allow_pickle=False)
        import tifffile

        boundaries = tifffile.imread(bnd_path)
        print("True-3D segmentation...", flush=True)
        seg_meta = run_true3d_seg(affinities, boundaries, seg_dir)
        labels = np.load(seg_dir / "labels.npy", allow_pickle=False)
        qc = machine_qc(affinities, labels, seg_meta, list(package_meta["shape_zyx"]))
        (seg_dir / "receipt.json").write_text(
            json.dumps(
                {
                    "created_at": _now(),
                    "segmentation": seg_meta,
                    "machine_qc": qc,
                    "qc_contract": "SEG_ROI_QC_CONTRACT_001",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        summary = {
            "id": "AFFINITY_ROI_001_SUMMARY",
            "created_at": _now(),
            "s4": s4["id"],
            "status": "AWAITING_HUMAN_S6_SEG_QC" if qc["verdict"] == "MACHINE_QC_PASS" else "MACHINE_QC_FAIL_ARCHIVED",
            "package": package_meta,
            "inference": inference_receipt,
            "segmentation": seg_meta,
            "machine_qc": qc,
            "full_volume_still_closed": True,
            "synapse_still_closed": True,
            "next_human_gate": "S6_HUMAN_SEG_QC_GATE"
            if qc["verdict"] == "MACHINE_QC_PASS"
            else "Diagnose ROI seg failure; do not full-volume",
        }
        (OUT / "ROI_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if FUNNEL_SUMMARY.exists():
            fs = json.loads(FUNNEL_SUMMARY.read_text(encoding="utf-8"))
            fs["status"] = summary["status"]
            fs["roi_opened"] = True
            fs["roi_summary"] = str((OUT / "ROI_SUMMARY.json").relative_to(REPO)).replace("\\", "/")
            fs["roi_still_closed"] = False
            fs["s6_pending"] = qc["verdict"] == "MACHINE_QC_PASS"
            FUNNEL_SUMMARY.write_text(json.dumps(fs, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(json.dumps({"status": summary["status"], "machine_qc": qc}, indent=2))
        return 0 if qc["verdict"] == "MACHINE_QC_PASS" else 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
