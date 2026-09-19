"""H1b/H3 causal comparison: does the non-equivariant review-unit geometry
produce materially different RAW contexts by axis, and do symmetric
counterfactual constructions remove the axis difference?

Read-only. Uses raw EM voxels keyed to the ACTUAL reviewed interface centres
(from the append-only event logs). Labels are used ONLY to stratify the final
report, never to construct any context statistic.

For every reviewed interface (by its center member, interface_member==1) we
compute raw-context statistics under three constructions of the review unit:

  A. HISTORICAL_ASYMMETRIC  - 3 members spread along other[0] only (as shipped)
  B. CENTER_ONLY            - the single affinity pair (fully axis-equivariant)
  C. SYMMETRIC_BOTH_ORTHO   - members spread equally along BOTH orthogonal axes

Statistic per construction (all axis-comparable, raw-only):
  * step        = |raw(right) - raw(left)| averaged over member pairs (the
                  affinity-edge intensity step; identical edge for A/B/C)
  * ctx_grad    = mean |3D gradient| over the voxels touched by the construction
  * ctx_std     = std of raw intensity over the voxels touched
The center affinity EDGE is identical across A/B/C by design (that is the point:
the edge is correct/equivariant; only the surrounding CONTEXT differs).
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/phase6e/MV-G3-AXIS-CONTEXT-COMPARISON-001.json"
AXES = {0: "Z", 1: "Y", 2: "X"}


def load_raw(path):
    return np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.float32)


def member_lefts(centre, axis, mode):
    z, y, x = centre
    other = [d for d in range(3) if d != axis]
    if mode == "CENTER_ONLY":
        offs = [(0, 0)]
    elif mode == "HISTORICAL_ASYMMETRIC":
        offs = [(0, 0), (-1, 0), (1, 0)]        # moves other[0] only
    elif mode == "SYMMETRIC_BOTH_ORTHO":
        offs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # both orthogonal axes
    else:
        raise ValueError(mode)
    lefts = []
    for o in offs:
        p = [z, y, x]
        p[other[0]] += o[0]
        p[other[1]] += o[1]
        lefts.append(tuple(p))
    return lefts


def in_bounds(p, shape, axis):
    r = list(p); r[axis] += 1
    return all(0 <= v < s for v, s in zip(p, shape)) and all(0 <= v < s for v, s in zip(r, shape))


def context_stats(raw, grad_mag, centre, axis, mode):
    lefts = member_lefts(centre, axis, mode)
    steps = []
    ctx_vals = []
    ctx_grads = []
    for left in lefts:
        if not in_bounds(left, raw.shape, axis):
            continue
        right = list(left); right[axis] += 1
        steps.append(abs(float(raw[tuple(right)]) - float(raw[tuple(left)])))
        ctx_vals.append(float(raw[left]))
        ctx_vals.append(float(raw[tuple(right)]))
        ctx_grads.append(float(grad_mag[left]))
        ctx_grads.append(float(grad_mag[tuple(right)]))
    if not steps:
        return None
    return {"step": float(np.mean(steps)), "ctx_grad": float(np.mean(ctx_grads)), "ctx_std": float(np.std(ctx_vals))}


def main():
    logs = glob.glob(str(REPO / "experiments/phase6e/g3-external-review-packages-*/MV-G3-*/workspace.events.jsonl"))
    # gather center members (interface_member==1) with their axis, decision, crop, split, raw path
    # need the raw path from the corresponding workspace.json
    ws_raw = {}
    for wf in glob.glob(str(REPO / "experiments/phase6e/g3-external-review-packages-*/MV-G3-*/workspace.json")):
        w = json.loads(Path(wf).read_text())
        ws_raw[w["id"]] = w["raw"]["path"]

    # decision per interface (member agreement)
    members_by_iface = collections.defaultdict(list)
    for lp in logs:
        for line in open(lp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            members_by_iface[(e["workspace_id"], e["crop_id"], e["split"], e["interface_id"])].append(e)

    raw_cache = {}
    grad_cache = {}
    # accumulate stats grouped by (mode, axis, decision)
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, members in members_by_iface.items():
        ws_id, crop, split, iface = key
        decisions = {m["decision"] for m in members}
        if len(decisions) != 1 or next(iter(decisions)) not in ("SAME_PROCESS", "DIFFERENT_PROCESS"):
            continue
        decision = next(iter(decisions))
        axis = int(members[0]["channel_zyx"])
        # center member is question_reference ending in '-01' (offset (0,0))
        def _member_idx(m):
            ref = str(m.get("question_reference", ""))
            try:
                return int(ref.split("-")[-1])
            except ValueError:
                return 99
        center = next((m for m in members if _member_idx(m) == 1), members[0])
        centre = tuple(center["pair_left_zyx"])
        raw_path = ws_raw[ws_id]
        if raw_path not in raw_cache:
            r = load_raw(raw_path)
            raw_cache[raw_path] = r
            grad_cache[raw_path] = np.linalg.norm(np.stack(np.gradient(r)), axis=0)
        raw, gmag = raw_cache[raw_path], grad_cache[raw_path]
        for mode in ("CENTER_ONLY", "HISTORICAL_ASYMMETRIC", "SYMMETRIC_BOTH_ORTHO"):
            st = context_stats(raw, gmag, centre, axis, mode)
            if st is None:
                continue
            for stat_name, val in st.items():
                acc[(mode, AXES[axis], stat_name)][decision].append(val)

    # summarize: for each (mode, stat) report per-axis mean, and axis spread
    def summarize(vals):
        return {"n": len(vals), "mean": float(np.mean(vals)) if vals else None, "std": float(np.std(vals)) if vals else None}

    report = {"id": "MV-G3-AXIS-CONTEXT-COMPARISON-001", "note": "raw-only; labels used only to stratify output", "constructions": {}}
    modes = ("CENTER_ONLY", "HISTORICAL_ASYMMETRIC", "SYMMETRIC_BOTH_ORTHO")
    stats = ("step", "ctx_grad", "ctx_std")
    for mode in modes:
        report["constructions"][mode] = {}
        for stat in stats:
            per_axis = {}
            axis_means = {}
            for ax in ("Z", "Y", "X"):
                allvals = acc[(mode, ax, stat)]["SAME_PROCESS"] + acc[(mode, ax, stat)]["DIFFERENT_PROCESS"]
                per_axis[ax] = {"all": summarize(allvals),
                                "SAME": summarize(acc[(mode, ax, stat)]["SAME_PROCESS"]),
                                "DIFFERENT": summarize(acc[(mode, ax, stat)]["DIFFERENT_PROCESS"])}
                axis_means[ax] = per_axis[ax]["all"]["mean"]
            means = [m for m in axis_means.values() if m is not None]
            axis_spread = (max(means) - min(means)) if means else None
            report["constructions"][mode][stat] = {"per_axis": per_axis, "axis_mean_spread": axis_spread}

    # headline: does axis_mean_spread of ctx_grad/ctx_std shrink from historical->symmetric->center?
    def spread(mode, stat):
        return report["constructions"][mode][stat]["axis_mean_spread"]
    report["headline"] = {
        stat: {mode: spread(mode, stat) for mode in modes} for stat in stats
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["headline"], indent=2))


if __name__ == "__main__":
    main()
