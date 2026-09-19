"""Build AFFINITY_S7_THROUGHPUT_PILOT_RECEIPT_001 from first-chunk logs + artifacts.

Run after MV-CHUNK-10019646 finishes (or with --partial for live estimate).
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/phase6e/AFFINITY-FULLVOL-S7-001"
CHUNK = "MV-CHUNK-10019646"
ENGINEERING = REPO / "experiments/phase6e/AFFINITY_S7_THROUGHPUT_ENGINEERING_001.json"
RECEIPT = REPO / "experiments/phase6e/AFFINITY_S7_THROUGHPUT_PILOT_RECEIPT_001.json"
N_CHUNKS = 28798
TILES_TOTAL = 20736


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_terminal(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    # header timestamps
    started = re.search(r"started_at: (.+)", text)
    running = re.search(r"running_for_ms: (\d+)", text)
    ended = re.search(r"ended_at: (.+)", text)
    exit_code = re.search(r"exit_code: (\d+)", text)
    tiles = [int(m.group(1)) for m in re.finditer(r"infer (\d+)/20736", text)]
    last_tile = tiles[-1] if tiles else 0
    shape = re.search(r"shape=\((\d+), (\d+), (\d+)\)", text)
    return {
        "started_at": started.group(1).strip() if started else None,
        "ended_at": ended.group(1).strip() if ended else None,
        "running_for_ms": int(running.group(1)) if running else None,
        "exit_code": int(exit_code.group(1)) if exit_code else None,
        "last_tile": last_tile,
        "tiles_total": TILES_TOTAL,
        "read_shape_zyx": [int(shape.group(1)), int(shape.group(2)), int(shape.group(3))] if shape else None,
        "completed": "segment " in text or "n_done" in text or (last_tile >= TILES_TOTAL),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terminal", type=Path, required=True)
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()

    eng = json.loads(ENGINEERING.read_text(encoding="utf-8"))
    log = parse_terminal(args.terminal)
    chunk_dir = OUT / "chunks" / CHUNK
    output_bytes = {}
    if chunk_dir.exists():
        output_bytes = {p.name: p.stat().st_size for p in chunk_dir.iterdir() if p.is_file()}

    # Prefer on-disk telemetry if worker wrote it (post-instrumentation runs)
    disk_tel = None
    tel_path = chunk_dir / "throughput_telemetry.json"
    if tel_path.exists():
        disk_tel = json.loads(tel_path.read_text(encoding="utf-8"))

    wall_s = None
    if log["running_for_ms"] is not None:
        wall_s = log["running_for_ms"] / 1000.0
    tiles_done = log["last_tile"]
    tps = (tiles_done / wall_s) if wall_s and tiles_done else None
    # Extrapolate full chunk infer if partial
    eta_chunk_s = (TILES_TOTAL / tps) if tps else None
    serial_hours = (N_CHUNKS * eta_chunk_s / 3600.0) if eta_chunk_s else None

    if disk_tel and disk_tel.get("infer_seconds"):
        eta_chunk_s = disk_tel["infer_seconds"]
        tps = disk_tel.get("tiles_per_second")
        serial_hours = N_CHUNKS * float(eta_chunk_s) / 3600.0

    if RECEIPT.exists() and not args.partial:
        raise FileExistsError(RECEIPT)

    receipt = {
        "id": "AFFINITY_S7_THROUGHPUT_PILOT_RECEIPT_001",
        "created_at": _now(),
        "status": "PARTIAL_LIVE_ESTIMATE" if args.partial or not log["completed"] else "COMPLETE",
        "engineering_id": eng["id"],
        "pilot_chunk_id": CHUNK,
        "from_terminal": str(args.terminal),
        "log_parse": log,
        "disk_telemetry": disk_tel,
        "output_bytes": output_bytes,
        "measured_or_estimated": {
            "tiles_done": tiles_done,
            "tiles_total": TILES_TOTAL,
            "fraction_complete": tiles_done / TILES_TOTAL,
            "wall_seconds_so_far": wall_s,
            "tiles_per_second_wall_proxy": tps,
            "estimated_infer_seconds_per_chunk": eta_chunk_s,
            "naive_serial_full_volume_gpu_hours": serial_hours,
            "naive_serial_full_volume_gpu_days": (serial_hours / 24.0) if serial_hours else None,
        },
        "gpu_utilization": "UNAVAILABLE_IN_WSL_ROOT_ROCM_SMI_DRIVER_NOT_INITIALIZED",
        "gates_unchanged": eng["gates_unchanged"],
        "next_step": "On COMPLETE receipt, freeze ONE throughput lever contract (T1–T6) citing this receipt",
    }
    out = RECEIPT if not args.partial else OUT / "THROUGHPUT_PILOT_PARTIAL.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt["measured_or_estimated"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
