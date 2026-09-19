"""VoxScout CLI ? analyze volumes and write receipts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parents[1]
for p in (REPO, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="voxscout", description="VoxScout spatial planner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("analyze", "run"):
        p = sub.add_parser(name, help="Analyze a volume (npy) or synthetic demo")
        p.add_argument("--volume", type=Path, default=None)
        p.add_argument("--prior-mask", type=Path, default=None)
        p.add_argument("--synthetic", action="store_true")
        p.add_argument("--out", type=Path, default=None)
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(func=_analyze)
    args = ap.parse_args(argv)
    return int(args.func(args))


def _synthetic_volume(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = np.zeros((40, 80, 80), dtype=np.uint8)
    vol[0:20, 0:40, 0:40] = 3
    vol[0:20, 40:80, 0:40] = rng.integers(40, 180, size=(20, 40, 40), dtype=np.uint8)
    vol[20:40, 0:40, 40:80] = rng.choice([0, 255], size=(20, 40, 40)).astype(np.uint8)
    vol[20:40, 40:80, 40:80] = 110
    vol[5:15, 50:70, 10:30] = 200
    return vol


def _analyze(args: argparse.Namespace) -> int:
    from voxscout._core_compat import load_core
    from voxscout.scout import VoxScout
    from voxscout.__version__ import __version__

    _, _, write_tool_receipt, ArtifactStore, ProvenanceGraph = load_core()

    volume = np.load(args.volume) if args.volume is not None else _synthetic_volume(args.seed)
    prior = np.load(args.prior_mask) if args.prior_mask is not None else None
    pmap = VoxScout().analyze(volume, prior_mask=prior)

    receipts_root = REPO / "receipts" / "voxscout"
    saved = pmap.save(args.out or (receipts_root / "priority_map.json"))
    hints = pmap.as_axonforge_hints()

    store = ArtifactStore(REPO)
    ref = store.put_json(hints, kind="voxscout_hints", name="axonforge_hints.json")

    prov = ProvenanceGraph()
    prov.add_node("volume", kind="volume", attrs={"shape": list(volume.shape), "dtype": str(volume.dtype)})
    prov.add_node("priority_map", kind="artifact", attrs={"path": str(saved)})
    if hasattr(prov, "record_edge"):
        prov.record_edge("volume", "priority_map", "analyzed_by_voxscout")

    summary = {
        "tool": "voxscout",
        "version": __version__,
        "volume_shape": list(volume.shape),
        "n_tiles": len(pmap.tiles),
        "class_counts": pmap.class_counts(),
        "priority_map": str(saved),
        "optimize_ok_tiles": sum(1 for h in hints.values() if h["optimize_ok"]),
        "hints_sha256": getattr(ref, "sha256", None),
        "provenance": prov.to_dict() if hasattr(prov, "to_dict") else prov.as_dict(),
    }
    receipt = write_tool_receipt("voxscout", summary)
    summary["receipt"] = str(receipt)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
