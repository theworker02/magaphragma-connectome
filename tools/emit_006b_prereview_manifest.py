"""Emit a concise machine-readable pre-review manifest for the -006b axis-neutral
paired experiment: per physical location -> crop, contrast stratum, three edge
coordinates, exclusion verdict, and hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from build_g3_axis_neutral_paired_experiment import historical_edges, three_edges  # noqa

Q_ROOT = REPO / "experiments/phase6e/g3-interface-queues-006b"
WS_ROOT = REPO / "experiments/phase6e/g3-external-review-packages-006b"
OUT = REPO / "experiments/phase6e/MV-G3-AXNEU2-PREREVIEW-MANIFEST-001.json"
CROPS = ["MV-G3-AXNEU2-LO1", "MV-G3-AXNEU2-LO2", "MV-G3-AXNEU2-HI1", "MV-G3-AXNEU2-HI2"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    hist = historical_edges()
    locations = []
    for crop in CROPS:
        ws = json.loads((WS_ROOT / crop / "workspace.json").read_text())
        q = json.loads((Q_ROOT / f"{crop}.json").read_text())
        raw = Path(ws["raw"]["path"])
        for loc in q["locations"]:
            c = tuple(loc["center_zyx"])
            edges = three_edges(c)
            edge_recs = []
            for (left, right, axis) in edges:
                key = (left, right, axis)
                edge_recs.append({"axis": ["Z", "Y", "X"][axis], "channel_zyx": axis,
                                  "pair_left_zyx": list(left), "pair_right_zyx": list(right),
                                  "excluded_historical": key in hist})
            locations.append({
                "location_id": f"{crop}:{loc['center_zyx']}",
                "crop_id": crop, "source_id": ws["parent_region_id"],
                "contrast_regime": ws["provenance"]["contrast_regime"],
                "contrast_band": loc["contrast_band"], "local_contrast": loc["contrast"],
                "center_zyx": loc["center_zyx"], "edges": edge_recs,
                "any_edge_excluded": any(e["excluded_historical"] for e in edge_recs),
                "raw_sha256": ws["raw"]["sha256"],
            })
    manifest = {
        "id": "MV-G3-AXNEU2-PREREVIEW-MANIFEST-001",
        "protocol": "MV-G3-AXIS-NEUTRAL-SAMPLING-PROTOCOL-002",
        "total_locations": len(locations), "total_edges": 3 * len(locations),
        "axis_balance": {"Z": len(locations), "Y": len(locations), "X": len(locations)},
        "any_location_with_excluded_edge": any(l["any_edge_excluded"] for l in locations),
        "crop_queue_hashes": {crop: _sha(Q_ROOT / f"{crop}.json") for crop in CROPS},
        "locations": locations,
    }
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT.resolve()), "locations": len(locations), "edges": 3 * len(locations),
                      "any_excluded": manifest["any_location_with_excluded_edge"],
                      "by_regime": {c: sum(1 for l in locations if l["crop_id"] == c) for c in CROPS}}, indent=2))


if __name__ == "__main__":
    main()
