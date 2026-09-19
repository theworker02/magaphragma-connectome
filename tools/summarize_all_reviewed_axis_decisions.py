"""Aggregate every reviewed Y/Z/X decision across all G3 external-review event
logs (-001..-004) by (split, axis, decision). Read-only characterization."""
from __future__ import annotations

import collections
import glob
import json

logs = glob.glob("experiments/phase6e/g3-external-review-packages-*/MV-G3-*/workspace.events.jsonl")
by_axis_split_class = collections.Counter()
by_source = collections.Counter()
for lp in logs:
    source = lp.split("g3-external-review-packages-")[1].split("\\")[0].split("/")[0]
    for line in open(lp, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        key = (e.get("split"), e.get("channel_name"), e.get("decision"))
        by_axis_split_class[key] += 1
        by_source[(source, e.get("channel_name"), e.get("decision"))] += 1

print("=== By (split, axis, decision) across ALL logs ===")
for k in sorted(by_axis_split_class, key=lambda t: tuple(str(x) for x in t)):
    print(f"  {k}: {by_axis_split_class[k]}")

print("\n=== By (review-tier, axis, decision) ===")
for k in sorted(by_source, key=lambda t: tuple(str(x) for x in t)):
    print(f"  {k}: {by_source[k]}")

# Y-specific TRAIN summary
y_train_same = by_axis_split_class[("G3_TARGET_TRAIN", "Y", "SAME_PROCESS")]
y_train_diff = by_axis_split_class[("G3_TARGET_TRAIN", "Y", "DIFFERENT_PROCESS")]
print(f"\nY x TRAIN: SAME={y_train_same}  DIFFERENT={y_train_diff}")
