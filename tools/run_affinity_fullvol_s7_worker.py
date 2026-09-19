"""S7 chunked full-volume affinity→seg worker (AFFINITY_FULL_VOLUME_S7_CONTRACT_001).

Resumable. Processes production-domain chunks with the authorized C checkpoint.
"""
from __future__ import annotations

import argparse
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
CONTRACT = REPO / "experiments/phase6e/AFFINITY_FULL_VOLUME_S7_CONTRACT_001.json"
S6 = REPO / "experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json"
DOMAIN = REPO / "local_research_build/phase5c-production/domain.json"
CHUNKS = REPO / "local_research_build/phase5c-production/chunks.json"
ROI_PKG = REPO / "experiments/phase6e/AFFINITY-ROI-001/package/ROI_PACKAGE.json"
OUT = REPO / "experiments/phase6e/AFFINITY-FULLVOL-S7-001"
CKPT = REPO / "experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt"
BASE = "https://waspem-dvid2.flatironinstitute.org/api/node/aa49d16e88424cdbae89f4b8ced2a49b/five_yuri_4contrast/raw/0_1_2/"
CROP_SIZE = (20, 64, 64)
STRIDE = (10, 32, 32)
STATE_PATH = OUT / "queue_state.json"


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
        return Path(f"{parts[2].upper()}:/" + "/".join(parts[3:]))
    if len(text) >= 2 and text[1] == ":" and on_linux:
        return Path(f"/mnt/{text[0].lower()}/{text[3:]}")
    return Path(text)


def overlaps(a: dict, b: dict) -> bool:
    for ax in ("x", "y", "z"):
        if a[ax][1] <= b[ax][0] or b[ax][1] <= a[ax][0]:
            return False
    return True


def roi_bounds_xyz(pkg: dict) -> dict:
    ox, oy, oz = pkg["roi_origin_xyz"]
    sx, sy, sz = pkg["roi_size_xyz"]
    return {"x": [ox, ox + sx], "y": [oy, oy + sy], "z": [oz, oz + sz]}


def load_state(chunks: list[dict]) -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "created_at": _now(),
        "contract_id": "AFFINITY_FULL_VOLUME_S7_CONTRACT_001",
        "completed": [],
        "failed": [],
        "status_by_id": {c["id"]: "NOT_STARTED" for c in chunks},
    }


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(STATE_PATH)


def priority_queue(chunks: list[dict], state: dict, roi: dict) -> list[dict]:
    by_id = {c["id"]: c for c in chunks}
    pending = [c for c in chunks if state["status_by_id"].get(c["id"]) == "NOT_STARTED"]
    seed = [c for c in pending if overlaps(c["core_bounds_xyz"], roi)]
    if not seed:
        seed = [c for c in pending if overlaps(c["read_bounds_xyz"], roi)]
    ordered = []
    seen = set()
    frontier = [c["id"] for c in sorted(seed, key=lambda c: c["id"])]
    while frontier:
        cid = frontier.pop(0)
        if cid in seen or state["status_by_id"].get(cid) != "NOT_STARTED":
            continue
        seen.add(cid)
        ordered.append(by_id[cid])
        for nid in by_id[cid].get("neighbor_chunk_ids", []):
            if nid not in seen and state["status_by_id"].get(nid) == "NOT_STARTED":
                frontier.append(nid)
    rest = sorted([c for c in pending if c["id"] not in seen], key=lambda c: c["id"])
    return ordered + rest


def fetch_raw(bounds: dict) -> tuple[np.ndarray, str]:
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    z0, z1 = bounds["z"]
    sx, sy, sz = x1 - x0, y1 - y0, z1 - z0
    url = BASE + f"{sx}_{sy}_{sz}/{x0}_{y0}_{z0}"
    expected = sx * sy * sz
    with urlopen(url, timeout=600) as response:
        payload = response.read()
    if len(payload) != expected:
        raise RuntimeError(f"bad payload {len(payload)} != {expected}")
    return np.frombuffer(payload, dtype=np.uint8).reshape((sz, sy, sx)), url


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
    import time

    import torch

    padding, starts = tile_layout(volume.shape)
    padded = np.pad(volume, padding, mode="reflect")
    sums = np.zeros((4,) + padded.shape, dtype=np.float32)
    weights = np.zeros(padded.shape, dtype=np.float32)
    patch_weight = gaussian_weight()
    total = len(starts[0]) * len(starts[1]) * len(starts[2])
    model = model.to(device).eval()
    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for index, position in enumerate(product(*starts), 1):
            region = tuple(slice(start, start + size) for start, size in zip(position, CROP_SIZE))
            patch = np.ascontiguousarray(padded[region], dtype=np.float32) / 255.0
            tensor = torch.from_numpy(patch[None, None]).to(device)
            affinities, boundaries = model(tensor)
            prediction = torch.cat((affinities, boundaries), dim=1)[0].float().cpu().numpy()
            sums[(slice(None),) + region] += prediction * patch_weight
            weights[region] += patch_weight
            if index % 200 == 0 or index == total:
                elapsed = time.perf_counter() - t0
                tps = index / elapsed if elapsed > 0 else float("nan")
                print(f"    infer {index}/{total} tiles_per_sec={tps:.3f}", flush=True)
    infer_seconds = time.perf_counter() - t0
    sums /= weights[None]
    np.clip(sums, 0.0, 1.0, out=sums)
    original = tuple(slice(pad[0], pad[0] + size) for pad, size in zip(padding, volume.shape))
    vram_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    telemetry = {
        "tiles_total": total,
        "infer_seconds": infer_seconds,
        "tiles_per_second": (total / infer_seconds) if infer_seconds > 0 else None,
        "vram_peak_bytes": vram_peak,
        "read_shape_zyx": list(volume.shape),
    }
    return sums[(slice(0, 3),) + original].copy(), sums[(3,) + original].copy(), telemetry


def crop_to_core(arr, read_bounds, core_bounds, is_czyx: bool = False):
    # arr is ZYX or CZYX over read_bounds
    oz = core_bounds["z"][0] - read_bounds["z"][0]
    oy = core_bounds["y"][0] - read_bounds["y"][0]
    ox = core_bounds["x"][0] - read_bounds["x"][0]
    dz = core_bounds["z"][1] - core_bounds["z"][0]
    dy = core_bounds["y"][1] - core_bounds["y"][0]
    dx = core_bounds["x"][1] - core_bounds["x"][0]
    if is_czyx:
        return arr[:, oz : oz + dz, oy : oy + dy, ox : ox + dx].copy()
    return arr[oz : oz + dz, oy : oy + dy, ox : ox + dx].copy()


def run_true3d_seg(affinities: np.ndarray, boundaries: np.ndarray, out_dir: Path, beta: float = 0.1) -> dict:
    import time

    import elf.segmentation.features as features
    import elf.segmentation.multicut as multicut
    import elf.segmentation.watershed as watershed

    t0 = time.perf_counter()
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
        "fragment_count": int(np.unique(fragments).size),
        "instance_count": int(np.unique(labels).size),
        "rag_edges": int(rag.numberOfEdges),
        "warnings": [str(w.message) for w in caught],
        "beta": beta,
        "segmentation_seconds": time.perf_counter() - t0,
    }


def process_chunk(chunk: dict, model, device: str) -> dict:
    import time

    cid = chunk["id"]
    chunk_dir = OUT / "chunks" / cid
    if chunk_dir.exists():
        raise FileExistsError(chunk_dir)
    chunk_dir.mkdir(parents=True)
    phases: dict[str, float] = {}
    wall0 = time.perf_counter()
    print(f"  fetch {cid} read_bounds={chunk['read_bounds_xyz']}", flush=True)
    t = time.perf_counter()
    raw, url = fetch_raw(chunk["read_bounds_xyz"])
    phases["dvid_fetch_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    np.save(chunk_dir / "raw_read_zyx.npy", raw, allow_pickle=False)
    phases["raw_save_seconds"] = time.perf_counter() - t
    print(f"  infer {cid} shape={raw.shape}", flush=True)
    affinities, boundaries, infer_tel = infer_volume(model, raw, device=device)
    phases["affinity_infer_seconds"] = float(infer_tel["infer_seconds"])
    t = time.perf_counter()
    aff_core = crop_to_core(affinities, chunk["read_bounds_xyz"], chunk["core_bounds_xyz"], is_czyx=True)
    bnd_core = crop_to_core(boundaries, chunk["read_bounds_xyz"], chunk["core_bounds_xyz"], is_czyx=False)
    np.save(chunk_dir / "affinities_core_czyx.npy", aff_core, allow_pickle=False)
    import tifffile

    tifffile.imwrite(chunk_dir / "boundaries_core.tif", bnd_core)
    phases["core_crop_save_seconds"] = time.perf_counter() - t
    del affinities, boundaries, raw
    print(f"  segment {cid} core={aff_core.shape[1:]}", flush=True)
    try:
        seg = run_true3d_seg(aff_core, bnd_core, chunk_dir)
        status = "COMPLETE"
        phases["segmentation_seconds"] = float(seg.get("segmentation_seconds") or 0.0)
    except ImportError:
        seg = {"error": "elf_unavailable_in_this_python"}
        status = "AFFINITY_DONE_SEG_PENDING"
        phases["segmentation_seconds"] = 0.0
    phases["total_wall_seconds"] = time.perf_counter() - wall0
    output_bytes = {
        p.name: p.stat().st_size
        for p in chunk_dir.iterdir()
        if p.is_file()
    }
    receipt = {
        "chunk_id": cid,
        "created_at": _now(),
        "source_url": url,
        "core_bounds_xyz": chunk["core_bounds_xyz"],
        "read_bounds_xyz": chunk["read_bounds_xyz"],
        "affinities_shape_czyx": list(aff_core.shape),
        "segmentation": seg,
        "status": status,
        "classification": "MACHINE_PSEUDOLABEL_CHUNK",
        "checkpoint_sha256": sha256(win_to_wsl(CKPT)),
        "throughput_telemetry": {
            **infer_tel,
            "phase_seconds": phases,
            "output_bytes": output_bytes,
            "io_wait_proxy_fetch_over_infer": (
                phases["dvid_fetch_seconds"] / phases["affinity_infer_seconds"]
                if phases["affinity_infer_seconds"] > 0
                else None
            ),
            "naive_serial_full_volume_gpu_hours": (
                28798 * phases["affinity_infer_seconds"] / 3600.0
                if phases["affinity_infer_seconds"] > 0
                else None
            ),
        },
        "throughput_engineering_id": "AFFINITY_S7_THROUGHPUT_ENGINEERING_001",
    }
    (chunk_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (chunk_dir / "throughput_telemetry.json").write_text(
        json.dumps(receipt["throughput_telemetry"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=1)
    ap.add_argument("--phase", choices=["infer_seg", "infer_only"], default="infer_only")
    args = ap.parse_args()

    s6 = json.loads(S6.read_text(encoding="utf-8"))
    if s6.get("status") != "APPROVED":
        raise SystemExit("S6 not approved")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    domain = json.loads(DOMAIN.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))["chunks"]
    roi = roi_bounds_xyz(json.loads(ROI_PKG.read_text(encoding="utf-8")))

    OUT.mkdir(parents=True, exist_ok=True)
    state = load_state(chunks)
    state["opened_at"] = state.get("opened_at") or _now()
    state["domain_id"] = domain["id"]
    state["contract_id"] = contract["id"]
    save_state(state)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm required")
    sys.path.insert(0, str(REPO / "third_party/segneuron/Train_and_Inference"))
    from model.Mnet import MNet

    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub")
    ckpt = torch.load(win_to_wsl(CKPT), map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_weights"], strict=True)

    queue = priority_queue(chunks, state, roi)
    print(f"queue pending={len(queue)} max={args.max_chunks}", flush=True)
    done = []
    for chunk in queue[: args.max_chunks]:
        cid = chunk["id"]
        state["status_by_id"][cid] = "RUNNING"
        save_state(state)
        try:
            receipt = process_chunk(chunk, model, device="cuda")
            state["status_by_id"][cid] = receipt["status"]
            state["completed"].append(cid)
            done.append(receipt)
        except Exception as exc:  # noqa: BLE001 — durable queue must continue
            state["status_by_id"][cid] = "FAILED"
            state["failed"].append({"id": cid, "error": str(exc)})
            print(f"FAILED {cid}: {exc}", flush=True)
        save_state(state)

    n_done = sum(1 for s in state["status_by_id"].values() if s in {"COMPLETE", "AFFINITY_DONE_SEG_PENDING"})
    n_fail = sum(1 for s in state["status_by_id"].values() if s == "FAILED")
    n_left = sum(1 for s in state["status_by_id"].values() if s == "NOT_STARTED")
    summary = {
        "id": "AFFINITY_FULLVOL_S7_001_PROGRESS",
        "updated_at": _now(),
        "chunks_completed_this_run": [r["chunk_id"] for r in done],
        "n_done": n_done,
        "n_failed": n_fail,
        "n_not_started": n_left,
        "n_total": len(chunks),
        "whole_domain_processed": n_left == 0 and n_fail == 0,
        "status": "RUNNING" if n_left else ("COMPLETE" if n_fail == 0 else "COMPLETE_WITH_FAILURES"),
    }
    (OUT / "PROGRESS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
