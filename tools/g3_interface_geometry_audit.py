"""Read-only geometry/semantics audit for the G3 interface construction.

Produces a machine-readable report covering:
  1. voxel spacing (isotropy) from the survey source metadata;
  2. affinity-edge equivariance (Property A) verdict;
  3. full-interface equivariance (Property B) exhaustive failure table;
  4. spread-axis source signature table;
  5. raw-data contextual comparison: for the actual historical reviewed
     interfaces, compare the current one-orthogonal-axis spread against
     axis-symmetric counterfactual neighborhoods, using raw-EM statistics only.
No labels are read; no historical artifact or generator is modified.
"""
from __future__ import annotations

import glob
import itertools
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SURVEY_MANIFEST = REPO / "local_research_build/phase6e/dvid_raw_context_survey_001/manifest.json"
OUT = REPO / "experiments/phase6e/MV-G3-INTERFACE-GEOMETRY-AUDIT-001.json"
AXES = {0: "Z", 1: "Y", 2: "X"}
PERMS = list(itertools.permutations((0, 1, 2)))


def current_members(centre, axis):
    other = [d for d in range(3) if d != axis]
    offsets = ((0, 0), (-1, 0), (1, 0))
    out = []
    for off in offsets:
        left = list(centre)
        left[other[0]] += off[0]
        left[other[1]] += off[1]
        out.append(tuple(left))
    return out


def symmetric_both_axes_members(centre, axis):
    """Axis-equivariant counterfactual: center + +/-1 along BOTH orthogonal axes."""
    other = [d for d in range(3) if d != axis]
    out = [tuple(centre)]
    for d in other:
        for delta in (-1, 1):
            c = list(centre)
            c[d] += delta
            out.append(tuple(c))
    return out  # 5 members, symmetric


def permute_coord(c, perm):
    out = [0, 0, 0]
    for a in range(3):
        out[perm[a]] = c[a]
    return tuple(out)


def inverse_perm(perm):
    inv = [0, 0, 0]
    for a in range(3):
        inv[perm[a]] = a
    return tuple(inv)


def edge(left, axis):
    right = left[:axis] + (left[axis] + 1,) + left[axis + 1:]
    return frozenset({tuple(left), tuple(right)})


def property_A(centre=(10, 20, 30)):
    for axis in (0, 1, 2):
        orig = edge(centre, axis)
        for perm in PERMS:
            pc = permute_coord(centre, perm)
            pe = edge(pc, perm[axis])
            inv = inverse_perm(perm)
            back = frozenset(permute_coord(v, inv) for v in pe)
            if back != orig:
                return False
    return True


def property_B_failures(centre=(10, 20, 30)):
    failures = []
    for axis in (0, 1, 2):
        orig_edges = {edge(m, axis) for m in current_members(centre, axis)}
        orig_offsets = sorted(tuple(m[d] - centre[d] for d in range(3)) for m in current_members(centre, axis))
        for perm in PERMS:
            pc = permute_coord(centre, perm)
            paxis = perm[axis]
            pmembers = current_members(pc, paxis)
            inv = inverse_perm(perm)
            back_edges = {frozenset(permute_coord(v, inv) for v in edge(m, paxis)) for m in pmembers}
            center_preserved = edge(centre, axis) in back_edges
            if back_edges != orig_edges:
                back_offsets = sorted(tuple(permute_coord(m, inv)[d] - centre[d] for d in range(3)) for m in pmembers)
                failures.append({
                    "original_affinity_axis": AXES[axis],
                    "permutation_orig_to_pos": list(perm),
                    "original_member_offsets": [list(o) for o in orig_offsets],
                    "inverse_mapped_member_offsets": [list(o) for o in back_offsets],
                    "center_edge_preserved": center_preserved,
                    "only_replicate_members_changed": center_preserved,
                })
    return failures


def spread_table(centre=(10, 20, 30)):
    table = {}
    for axis in (0, 1, 2):
        other = [d for d in range(3) if d != axis]
        moved = sorted({d for m in current_members(centre, axis) for d in range(3) if m[d] != centre[d]})
        table[AXES[axis]] = {
            "orthogonal_axes_available": [AXES[d] for d in other],
            "orthogonal_axis_used_for_spread": [AXES[d] for d in moved],
            "physical_member_offsets": [list(tuple(m[d] - centre[d] for d in range(3))) for m in current_members(centre, axis)],
        }
    return table


def _clip(c, shape):
    return all(0 <= c[d] < shape[d] for d in range(3))


def raw_context_comparison():
    """For each historical -005 reviewed interface centre, compute raw-EM context
    statistics for (a) current one-axis spread and (b) symmetric both-axis
    neighborhood, grouped by affinity axis. Uses raw voxels only."""
    manifest = json.loads(SURVEY_MANIFEST.read_text())
    src = {r["id"]: r for r in manifest["records"]}
    per_axis = {AXES[a]: {"current_spread_context_std": [], "symmetric_context_std": [],
                          "current_spread_grad": [], "symmetric_context_grad": []} for a in (0, 1, 2)}
    logs = glob.glob(str(REPO / "experiments/phase6e/g3-external-review-packages-005/MV-G3-*/workspace.json"))
    for wf in logs:
        w = json.loads(Path(wf).read_text())
        arr = np.asarray(np.load(w["raw"]["path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        shape = arr.shape
        qpath = REPO / "experiments/phase6e/g3-interface-queues-005" / f"{w['crop_id']}.json"
        q = json.loads(qpath.read_text())
        # center = interface_member 1
        for item in q["questions"]:
            if int(item["interface_member"]) != 1:
                continue
            axis = int(item["channel_zyx"])
            centre = tuple(item["pair_left_zyx"])
            cur = [c for c in current_members(centre, axis) if _clip(c, shape)]
            sym = [c for c in symmetric_both_axes_members(centre, axis) if _clip(c, shape)]
            cur_vals = np.array([arr[c] for c in cur], dtype=np.float32)
            sym_vals = np.array([arr[c] for c in sym], dtype=np.float32)
            per_axis[AXES[axis]]["current_spread_context_std"].append(float(cur_vals.std()))
            per_axis[AXES[axis]]["symmetric_context_std"].append(float(sym_vals.std()))
    summary = {}
    for ax, d in per_axis.items():
        summary[ax] = {
            "n_interfaces": len(d["current_spread_context_std"]),
            "current_spread_context_std_mean": float(np.mean(d["current_spread_context_std"])) if d["current_spread_context_std"] else None,
            "symmetric_context_std_mean": float(np.mean(d["symmetric_context_std"])) if d["symmetric_context_std"] else None,
        }
    return summary


def main():
    manifest = json.loads(SURVEY_MANIFEST.read_text())
    voxel = manifest["source"]["voxel_size_nm_xyz"]
    report = {
        "id": "MV-G3-INTERFACE-GEOMETRY-AUDIT-001",
        "read_only": True,
        "voxel_spacing_nm_xyz": voxel,
        "isotropic": len(set(voxel)) == 1,
        "one_voxel_physical_distance_nm": {"Z": voxel[2], "Y": voxel[1], "X": voxel[0]},
        "H2_anisotropy_supported": len(set(voxel)) != 1,
        "property_A_affinity_edge_equivariant": property_A(),
        "property_B_full_interface_failures": property_B_failures(),
        "spread_axis_source_signature": {
            "construction": "other=[d for d in range(3) if d!=axis]; offsets=((0,0),(-1,0),(1,0)); only other[0] moves => spread_axis=other[0]",
            "table": spread_table(),
        },
        "raw_context_comparison_005": raw_context_comparison(),
    }
    report["classification"] = {
        "H1a_pair_or_coord_bug": "NOT_SUPPORTED" if report["property_A_affinity_edge_equivariant"] else "SUPPORTED",
        "H1b_interface_context_geometric_asymmetry": "SUPPORTED" if report["property_B_full_interface_failures"] else "NOT_SUPPORTED",
        "H2_anisotropy": "NOT_SUPPORTED (isotropic 8nm)" if report["isotropic"] else "SUPPORTED",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("voxel_spacing_nm_xyz", "isotropic", "property_A_affinity_edge_equivariant", "classification")}, indent=2))
    print("spread table:", json.dumps(report["spread_axis_source_signature"]["table"], indent=2))
    print("raw context:", json.dumps(report["raw_context_comparison_005"], indent=2))
    print("full-interface failures:", len(report["property_B_full_interface_failures"]))


if __name__ == "__main__":
    main()
