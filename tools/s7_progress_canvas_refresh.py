#!/usr/bin/env python3
"""Regenerate the Affinity S7 progress Cursor canvas with a fresh snapshot.

Reads S7_OUT (default ~/s7-affinity-out) via s7_progress_snapshot, then rewrites
the DATA constant in affinity-s7-progress.canvas.tsx.

Usage:
  python tools/s7_progress_canvas_refresh.py
  python tools/s7_progress_canvas_refresh.py --out /home/research-runner/s7-affinity-out
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from s7_progress_snapshot import DEFAULT_CHUNKS, default_out, snapshot  # noqa: E402

# Cursor managed canvases directory for this workspace
CANVAS_CANDIDATES = [
    Path.home()
    / ".cursor"
    / "projects"
    / "c-Users-matth-OneDrive-Desktop-magaphragma-connectome"
    / "canvases"
    / "affinity-s7-progress.canvas.tsx",
    REPO / "canvases" / "affinity-s7-progress.canvas.tsx",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_canvas() -> Path:
    for p in CANVAS_CANDIDATES:
        if p.is_file():
            return p
    raise SystemExit(
        "canvas not found; expected "
        + str(CANVAS_CANDIDATES[0])
    )


def data_block(snap: dict, generated_at: str) -> str:
    remaining = max(0, int(snap["total"]) - int(snap["completed"]))
    payload = {
        "id": snap.get("id", "AFFINITY_S7_HYBRID_PROGRESS"),
        "title": snap.get("title", "Affinity S7"),
        "subtitle": "hybrid drain · local even + Vast odd",
        "generatedAt": generated_at,
        "snapshotAt": snap.get("updated_at", generated_at),
        "snapshotSource": "Fresh read via tools/s7_progress_canvas_refresh.py → s7_progress_snapshot.",
        "out": snap.get("out", ""),
        "total": int(snap["total"]),
        "completed": int(snap["completed"]),
        "pct": float(snap["pct"]),
        "local_even": int(snap["local_even"]),
        "vast_odd": int(snap["vast_odd"]),
        "remaining": remaining,
        "shards": {
            "local": {
                "id": 0,
                "label": "Shard 0 · Local · even",
                "role": "RX 7800 XT / ROCm",
                "done": int(snap["local_even"]),
            },
            "vast": {
                "id": 1,
                "label": "Shard 1 · Vast · odd",
                "role": "RTX 3060 Ti offload",
                "done": int(snap["vast_odd"]),
            },
        },
    }
    # Pretty TS object (JSON is valid enough for our literal shape).
    body = json.dumps(payload, indent=2)
    # quote keys stay double-quoted; convert to TS `as const` block
    return "const DATA = " + body + " as const;\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--canvas", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or default_out()
    snap = snapshot(out, args.chunks)
    generated = _now()
    canvas = args.canvas or find_canvas()
    text = canvas.read_text(encoding="utf-8")
    new_data = data_block(snap, generated)
    pat = re.compile(r"const DATA = \{.*?\n\} as const;\n", re.DOTALL)
    if not pat.search(text):
        raise SystemExit(f"DATA block not found in {canvas}")
    updated = pat.sub(new_data, text, count=1)
    canvas.write_text(updated, encoding="utf-8")
    print(
        json.dumps(
            {
                "canvas": str(canvas),
                "pct": snap["pct"],
                "completed": snap["completed"],
                "total": snap["total"],
                "local_even": snap["local_even"],
                "vast_odd": snap["vast_odd"],
                "generatedAt": generated,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
