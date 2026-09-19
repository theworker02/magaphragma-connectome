"""Read-only measurement of per-axis raw-EM directional statistics across all
G3 TRAIN crops (A-L). Discriminates H2 (anisotropy) / H3 (candidate-selection)
/ H4 (biological). Reads raw voxels only; no labels, no modification.

Reports per crop and pooled:
  * mean |directional gradient| along Z, Y, X (np.gradient components);
  * dominant-axis assignment fraction (argmax|grad|) over interior voxels;
  * dominant-axis fraction among the TOP-gradient tail (the candidates the
    interface generator actually selects);
  * physical voxel spacing per axis (from metadata).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SURVEY = json.loads((REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json").read_text())
SRC = {r["id"]: r for r in SURVEY["records"]}
VOXEL_NM_XYZ = SURVEY["source"]["voxel_size_nm_xyz"]  # [x,y,z]
MARGIN = 3
OUT = REPO / "experiments/phase6e/MV-G3-AXIS-GRADIENT-STATISTICS-001.json"


def crop_sources():
    """Map crop_id -> source_id from every G3 workspace (A-L)."""
    out = {}
    for wf in glob.glob(str(REPO / "experiments/phase6e/g3-external-review-packages-*/MV-G3-TRAIN-*/workspace.json")):
        w = json.loads(Path(wf).read_text())
        out[w["crop_id"]] = w["parent_region_id"]
    return out


def measure(arr):
    gz, gy, gx = np.gradient(arr)  # components along axis 0(Z),1(Y),2(X)
    interior = np.zeros(arr.shape, dtype=bool)
    interior[MARGIN:-MARGIN, MARGIN:-MARGIN, MARGIN:-MARGIN] = True
    absg = np.stack([np.abs(gz), np.abs(gy), np.abs(gx)])  # (3,Z,Y,X)
    mean_abs = [float(absg[a][interior].mean()) for a in range(3)]
    dominant = np.argmax(absg, axis=0)
    dom_int = dominant[interior]
    frac_all = [float(np.mean(dom_int == a)) for a in range(3)]
    # top tail: voxels above the 99th percentile of gradient magnitude
    score = np.linalg.norm(absg, axis=0)
    thr = np.quantile(score[interior], 0.99)
    tail = interior & (score >= thr)
    dom_tail = dominant[tail]
    frac_tail = [float(np.mean(dom_tail == a)) for a in range(3)] if dom_tail.size else [0, 0, 0]
    return {"mean_abs_grad_ZYX": mean_abs, "dominant_axis_fraction_all_ZYX": frac_all,
            "dominant_axis_fraction_top1pct_ZYX": frac_tail}


def main():
    cs = crop_sources()
    per_crop = {}
    pooled_mean = np.zeros(3)
    pooled_tail = np.zeros(3)
    n = 0
    for crop_id, src in sorted(cs.items()):
        rec = SRC[src]
        arr = np.asarray(np.load(rec["raw_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        m = measure(arr)
        per_crop[crop_id] = {"source_id": src, **m}
        pooled_mean += np.array(m["mean_abs_grad_ZYX"])
        pooled_tail += np.array(m["dominant_axis_fraction_top1pct_ZYX"])
        n += 1
    report = {
        "id": "MV-G3-AXIS-GRADIENT-STATISTICS-001",
        "voxel_size_nm_xyz": VOXEL_NM_XYZ,
        "voxel_isotropic": len(set(VOXEL_NM_XYZ)) == 1,
        "axis_index_legend": {"0": "Z", "1": "Y", "2": "X"},
        "per_crop": per_crop,
        "pooled_mean_abs_grad_ZYX": (pooled_mean / n).tolist(),
        "pooled_dominant_axis_fraction_top1pct_ZYX": (pooled_tail / n).tolist(),
        "crops_measured": n,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"voxel_size_nm_xyz": VOXEL_NM_XYZ, "isotropic": report["voxel_isotropic"],
                      "pooled_mean_abs_grad_ZYX": report["pooled_mean_abs_grad_ZYX"],
                      "pooled_dominant_axis_fraction_top1pct_ZYX": report["pooled_dominant_axis_fraction_top1pct_ZYX"],
                      "crops": n}, indent=2))


if __name__ == "__main__":
    main()
