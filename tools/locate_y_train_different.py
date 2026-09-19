"""Locate every Y-axis TRAIN DIFFERENT_PROCESS reviewed event and say which
event log / tier it lives in, and whether that workspace fed frozen -002."""
from __future__ import annotations

import glob
import json

frozen_manifest = json.load(open("experiments/phase6e/MV-G3-DVID-REVIEWED-INTERFACE-AFFINITY-002.json"))
frozen_crops = {r["crop_id"] for r in frozen_manifest["regions"]}

hits = []
for lp in glob.glob("experiments/phase6e/g3-external-review-packages-*/MV-G3-*/workspace.events.jsonl"):
    tier = lp.split("g3-external-review-packages-")[1].split("\\")[0].split("/")[0]
    for line in open(lp, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        if e.get("channel_name") == "Y" and e.get("split") == "G3_TARGET_TRAIN" and e.get("decision") == "DIFFERENT_PROCESS":
            hits.append({"tier": tier, "crop": e.get("crop_id"), "workspace_id": e.get("workspace_id"), "interface": e.get("interface_id"), "pair_left": e.get("pair_left_zyx")})

by_tier_crop = {}
for h in hits:
    by_tier_crop.setdefault((h["tier"], h["crop"]), 0)
    by_tier_crop[(h["tier"], h["crop"])] += 1

print(f"Total Y/TRAIN/DIFFERENT events found: {len(hits)}")
for (tier, crop), n in sorted(by_tier_crop.items()):
    print(f"  tier -{tier}  {crop}: {n}   (crop in frozen -002 manifest: {crop in frozen_crops})")

# Which frozen workspace ids does -002 actually bind?
bindings = json.load(open("experiments/phase6e/G3_002_INTERFACE_BINDINGS.json"))["regions"]
print("\nFrozen -002 binds these workspace files:")
for crop, b in bindings.items():
    print(f"  {crop}: {b['workspace']}")
