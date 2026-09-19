"""Read-only H3 mechanism audit: does argmax(|gradient|) + top-score selection
create a Z-vs-YX axis bias that is absent in the full population but severe in
the extreme tail?

For each TRAIN crop (A-H frozen sources + I-L new sources) and pooled:
  * per-axis gradient magnitude stats (mean/median/q75/q90/q95/q99) using the
    signed partial-derivative magnitude |d/daxis| per axis;
  * argmax(|gradient|) axis SHARE within: all voxels, top 25/10/5/1% by the
    combined gradient score (L2 norm), matching the generator's score;
  * separated-centre axis share (the generator's actual selectable set);
  * historical-review axis share (interfaces actually reviewed).

Does not modify the generator. Uses raw EM only.
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MARGIN = 3
SEP2 = 144
AXES = {0: "Z", 1: "Y", 2: "X"}
OUT = REPO / "experiments/phase6e/MV-G3-AXIS-GRADIENT-TAIL-AUDIT-001.json"

# Map crop_id -> raw source, from the survey manifest + existing workspaces.
def crop_sources():
    manifest = json.loads((REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json").read_text())
    src = {r["id"]: r for r in manifest["records"]}
    out = {}
    for wf in glob.glob(str(REPO / "experiments/phase6e/g3-external-review-packages-*/MV-G3-TRAIN-*/workspace.json")):
        w = json.loads(Path(wf).read_text())
        cid = w["crop_id"]
        if cid not in out:
            out[cid] = src[w["parent_region_id"]]["raw_path"]
    return out


def separated_centres(coords):
    grid = {}
    kept = []
    for zyx in coords:
        z, y, x = int(zyx[0]), int(zyx[1]), int(zyx[2])
        cell = (z // 12, y // 12, x // 12)
        conflict = False
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for (az, ay, ax) in grid.get((cell[0]+dz, cell[1]+dy, cell[2]+dx), ()):
                        if (az-z)**2+(ay-y)**2+(ax-x)**2 < SEP2:
                            conflict = True; break
                    if conflict: break
                if conflict: break
            if conflict: break
        if not conflict:
            grid.setdefault(cell, []).append((z, y, x)); kept.append((z, y, x))
    return kept


def measure_crop(raw_path):
    arr = np.asarray(np.load(raw_path, mmap_mode="r", allow_pickle=False), dtype=np.float32)
    grads = np.stack(np.gradient(arr))            # (3,Z,Y,X) partials d/dz,d/dy,d/dx
    absg = np.abs(grads)
    score = np.linalg.norm(grads, axis=0)
    interior = np.zeros(arr.shape, bool)
    interior[MARGIN:-MARGIN, MARGIN:-MARGIN, MARGIN:-MARGIN] = True
    dom = np.argmax(absg, axis=0)

    idx = interior
    per_axis_grad = {}
    for a in (0, 1, 2):
        v = absg[a][idx]
        per_axis_grad[AXES[a]] = {
            "mean": float(v.mean()), "median": float(np.median(v)),
            "q75": float(np.quantile(v, .75)), "q90": float(np.quantile(v, .90)),
            "q95": float(np.quantile(v, .95)), "q99": float(np.quantile(v, .99)),
        }

    dom_i = dom[idx]
    sc_i = score[idx]
    def share(mask):
        d = dom_i[mask]
        n = int(d.size)
        return {AXES[a]: (float(np.mean(d == a)) if n else None) for a in (0, 1, 2)} | {"n": n}
    order_thresholds = {}
    n = sc_i.size
    ranks = np.argsort(sc_i)[::-1]
    for label, frac in (("all", 1.0), ("top25", .25), ("top10", .10), ("top5", .05), ("top1", .01)):
        k = int(np.ceil(frac * n))
        m = np.zeros(n, bool); m[ranks[:k]] = True
        order_thresholds[label] = share(m)

    # separated centres axis share (over full interior population, all axes)
    coords = np.argwhere(interior)
    vals = score[interior]
    o = np.argsort(vals)[::-1]
    sep = separated_centres(coords[o])
    sep_axis = collections.Counter(int(dom[z, y, x]) for (z, y, x) in sep)
    sep_total = sum(sep_axis.values()) or 1
    sep_share = {AXES[a]: sep_axis[a] / sep_total for a in (0, 1, 2)} | {"n": sep_total}

    return {"per_axis_gradient": per_axis_grad, "argmax_share": order_thresholds, "separated_centre_share": sep_share}


def historical_review_share():
    logs = glob.glob(str(REPO / "experiments/phase6e/g3-external-review-packages-*/MV-G3-TRAIN-*/workspace.events.jsonl"))
    per_crop = collections.defaultdict(lambda: collections.Counter())
    for lp in logs:
        crop = Path(lp).parent.name
        for line in open(lp, encoding="utf-8"):
            line = line.strip()
            if not line: continue
            e = json.loads(line)
            per_crop[crop][e.get("channel_name")] += 1
    return {c: dict(v) for c, v in per_crop.items()}


def main():
    sources = crop_sources()
    per_crop = {}
    pooled_argmax = {lbl: collections.Counter() for lbl in ("all", "top25", "top10", "top5", "top1")}
    pooled_n = {lbl: 0 for lbl in pooled_argmax}
    for cid in sorted(sources):
        m = measure_crop(sources[cid])
        per_crop[cid] = m
        for lbl in pooled_argmax:
            s = m["argmax_share"][lbl]
            for a in ("Z", "Y", "X"):
                if s[a] is not None:
                    pooled_argmax[lbl][a] += s[a] * s["n"]
            pooled_n[lbl] += s["n"]
    pooled = {lbl: ({a: (pooled_argmax[lbl][a] / pooled_n[lbl]) for a in ("Z", "Y", "X")} | {"n": pooled_n[lbl]}) for lbl in pooled_argmax}
    report = {
        "id": "MV-G3-AXIS-GRADIENT-TAIL-AUDIT-001",
        "voxel_size_nm_xyz": [8, 8, 8],
        "note": "np.gradient index 0=Z,1=Y,2=X; score = L2 norm of the 3 partials (generator's score).",
        "per_crop": per_crop,
        "pooled_argmax_share": pooled,
        "historical_review_share": historical_review_share(),
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # concise stdout
    print(json.dumps({"pooled_argmax_share": pooled}, indent=2))


if __name__ == "__main__":
    main()
