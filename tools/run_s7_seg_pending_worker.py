#!/usr/bin/env python3
"""S8 candidate seg worker: ELF watershed+multicut on AFFINITY_DONE_SEG_PENDING chunks.

CPU/elf only — safe to run beside the GPU affinity drain. Machine pseudolabels only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/phase6e/AFFINITY-FULLVOL-S7-001"
STATE_PATH = OUT / "queue_state.json"
CONTRACT = REPO / "experiments/phase6e/AFFINITY_S8_CANDIDATE_SEG_CONTRACT_001.json"
ROI_PKG = REPO / "experiments/phase6e/AFFINITY-ROI-001/package/ROI_PACKAGE.json"
CHUNKS = REPO / "local_research_build/phase5c-production/chunks.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def overlaps(a: dict, b: dict) -> bool:
    for ax in ("x", "y", "z"):
        if a[ax][1] <= b[ax][0] or b[ax][1] <= a[ax][0]:
            return False
    return True


def roi_bounds_xyz(pkg: dict) -> dict:
    ox, oy, oz = pkg["roi_origin_xyz"]
    sx, sy, sz = pkg["roi_size_xyz"]
    return {"x": [ox, ox + sx], "y": [oy, oy + sy], "z": [oz, oz + sz]}


def run_true3d_seg(affinities: np.ndarray, boundaries: np.ndarray, out_dir: Path, beta: float = 0.1) -> dict:
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
        # current elf: (rag, segmentation, affinity_map|input, [offsets])
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


def priority_pending(chunks: list[dict], state: dict, roi: dict) -> list[dict]:
    by_id = {c["id"]: c for c in chunks}
    pending_ids = [
        cid
        for cid, st in state.get("status_by_id", {}).items()
        if st == "AFFINITY_DONE_SEG_PENDING"
        and (OUT / "chunks" / cid / "affinities_core_czyx.npy").exists()
        and not (OUT / "chunks" / cid / "labels.npy").exists()
    ]
    seed = [cid for cid in pending_ids if overlaps(by_id[cid]["core_bounds_xyz"], roi)]
    ordered, seen, frontier = [], set(), sorted(seed)
    while frontier:
        cid = frontier.pop(0)
        if cid in seen:
            continue
        seen.add(cid)
        ordered.append(by_id[cid])
        for nid in by_id[cid].get("neighbor_chunk_ids", []):
            if nid in pending_ids and nid not in seen:
                frontier.append(nid)
    rest = sorted([by_id[cid] for cid in pending_ids if cid not in seen], key=lambda c: c["id"])
    return ordered + rest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chunks", type=int, default=4)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--beta", type=float, default=0.1)
    args = ap.parse_args()

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN":
        raise SystemExit("S8 contract not FROZEN")
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))["chunks"]
    roi = roi_bounds_xyz(json.loads(ROI_PKG.read_text(encoding="utf-8")))

    # Prove elf before claiming work
    import elf.segmentation.watershed  # noqa: F401

    done_run = []
    while True:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        queue = priority_pending(chunks, state, roi)
        print(f"S8 seg pending={len(queue)} max={args.max_chunks}", flush=True)
        if not queue:
            break
        for chunk in queue[: args.max_chunks]:
            cid = chunk["id"]
            chunk_dir = OUT / "chunks" / cid
            print(f"  segment {cid}", flush=True)
            aff = np.load(chunk_dir / "affinities_core_czyx.npy")
            import tifffile

            bnd = tifffile.imread(chunk_dir / "boundaries_core.tif")
            try:
                seg = run_true3d_seg(aff, bnd, chunk_dir, beta=args.beta)
                status = "COMPLETE"
            except Exception as exc:  # noqa: BLE001
                seg = {"error": str(exc)}
                status = "AFFINITY_DONE_SEG_PENDING"
                print(f"  FAILED {cid}: {exc}", flush=True)
            receipt = {
                "chunk_id": cid,
                "created_at": _now(),
                "contract_id": contract["id"],
                "classification": "MACHINE_PSEUDOLABEL_CHUNK",
                "status": status,
                "segmentation": seg,
            }
            (chunk_dir / "seg_receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            state["status_by_id"][cid] = status
            if status == "COMPLETE":
                state.setdefault("completed", [])
                if cid not in state["completed"]:
                    state["completed"].append(cid)
                done_run.append(cid)
            STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"chunk": cid, "status": status, "seg": seg}, default=str), flush=True)

        n_complete = sum(1 for s in state["status_by_id"].values() if s == "COMPLETE")
        n_seg_pending = sum(1 for s in state["status_by_id"].values() if s == "AFFINITY_DONE_SEG_PENDING")
        progress = {
            "id": "AFFINITY_S8_SEG_PROGRESS",
            "updated_at": _now(),
            "n_complete": n_complete,
            "n_affinity_done_seg_pending": n_seg_pending,
            "chunks_completed_this_run": done_run[-50:],
        }
        (OUT / "S8_SEG_PROGRESS.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.loop or n_seg_pending == 0:
            if args.loop and n_seg_pending == 0:
                print("S8 idle — waiting for more AFFINITY_DONE_SEG_PENDING", flush=True)
                time.sleep(30)
                continue
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
