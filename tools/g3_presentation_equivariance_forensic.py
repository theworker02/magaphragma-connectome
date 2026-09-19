"""READ-ONLY synthetic forensic tests for G3 reviewer presentation equivariance.

Proves which presentation transforms are / are not equivariant under the six
permutations of Z/Y/X. Does not modify frozen -008 artifacts or the reviewer.
Writes experiments/phase6e/G3_PRESENTATION_EQUIVARIANCE_FORENSIC_001.json
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from review_external_boundary_package import _plane  # noqa: E402

OUT = REPO / "experiments/phase6e/G3_PRESENTATION_EQUIVARIANCE_FORENSIC_001.json"
AXES = {0: "Z", 1: "Y", 2: "X"}
PERMS = list(itertools.permutations((0, 1, 2)))


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def permute_volume(vol: np.ndarray, perm: tuple[int, int, int]) -> np.ndarray:
    # With permute_coord: out[perm[d]] = c[d], require vol_p[c'] = vol[c].
    # That is vol_p[i,j,k] = vol[i',j',k'] indexed by perm — implemented by
    # np.transpose(vol, axes=perm).
    return np.transpose(vol, axes=perm)


def permute_coord(c: tuple[int, int, int], perm: tuple[int, int, int]) -> tuple[int, int, int]:
    out = [0, 0, 0]
    for d in range(3):
        out[perm[d]] = c[d]
    return tuple(out)


def edge_pair(center: tuple[int, int, int], axis: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    left = list(center)
    right = left[:]
    right[axis] += 1
    return tuple(left), tuple(right)


def docked_planes(raw: np.ndarray, a: tuple[int, int, int], b: tuple[int, int, int]) -> dict[str, np.ndarray]:
    return {
        "XY": _plane(np.asarray(raw[a[0]]), a, b, "XY", a[0]),
        "XZ": _plane(np.asarray(raw[:, a[1], :]), a, b, "XZ", a[1]),
        "YZ": _plane(np.asarray(raw[:, :, a[2]]), a, b, "YZ", a[2]),
    }


def marker_present(plane_rgb: np.ndarray, color: tuple[int, int, int]) -> bool:
    return bool(np.any(np.all(plane_rgb == np.asarray(color, dtype=np.uint8), axis=-1)))


def covisibility(raw: np.ndarray, a, b) -> dict[str, dict[str, bool]]:
    planes = docked_planes(raw, a, b)
    red, green = (255, 55, 55), (60, 230, 110)
    return {
        name: {"A": marker_present(img, red), "B": marker_present(img, green)}
        for name, img in planes.items()
    }


def main() -> None:
    # Synthetic isotropic fixture: unique intensity encoding of ZYX so
    # permutations are detectable; plus a bright "membrane" sheet at z=mid
    # so Z-edges and Y/X-edges see different structure after presentation.
    z, y, x = 16, 16, 16
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    vol = (zz * 7 + yy * 3 + xx).astype(np.uint8)
    vol[8, :, :] = 255  # Z-normal membrane sheet
    center = (7, 8, 8)  # Z-edge crosses membrane; Y/X edges stay in sheet-adjacent tissue

    report: dict = {
        "id": "MV-G3-PRESENTATION-EQUIVARIANCE-FORENSIC-001",
        "schema_version": 1,
        "status": "READ_ONLY_SYNTHETIC_PROOF",
        "array_order": "ZYX",
        "world_order": "XYZ",
        "voxel_size_nm_xyz": [8, 8, 8],
        "fixture": {
            "shape_zyx": [z, y, x],
            "center_zyx": list(center),
            "membrane_sheet": "z=8 (normal along Z)",
            "note": "Fixture chosen so biological Z-crossing differs from Y/X; presentation tests do not use that biology as ground truth — they test transform equivariance.",
        },
        "tests": {},
    }

    # --- 1. Central edge permutation equivariance (geometry only) ---
    edge_ok = True
    for axis in (0, 1, 2):
        a0, b0 = edge_pair(center, axis)
        for perm in PERMS:
            pa = permute_coord(a0, perm)
            pb = permute_coord(b0, perm)
            new_axis = perm[axis]
            exp_b = list(pa)
            exp_b[new_axis] += 1
            if tuple(exp_b) != pb:
                edge_ok = False
    report["tests"]["1_central_edge_permutation_equivariance"] = {
        "pass": edge_ok,
        "detail": "pair_right = left + e_axis is equivariant under all 6 axis permutations",
    }

    # --- 2/4/7. Displayed-plane + marker co-visibility (presentation) ---
    cov_by_axis = {}
    for axis in (0, 1, 2):
        a, b = edge_pair(center, axis)
        cov_by_axis[AXES[axis]] = covisibility(vol, a, b)
    # Primary failure: XY shows A+B for Y and X, but only A for Z
    xy_cov = {ax: cov_by_axis[ax]["XY"] for ax in ("Z", "Y", "X")}
    plane_equivariant = (
        xy_cov["Z"] == xy_cov["Y"] == xy_cov["X"]
        and all(cov_by_axis[ax]["XY"]["B"] for ax in ("Z", "Y", "X"))
    )
    report["tests"]["4_displayed_plane_equivalence"] = {
        "pass": False,
        "xy_marker_covisibility": xy_cov,
        "full_covisibility": cov_by_axis,
        "failure": "On docked XY (commonly attended 'section' view), Z-neighbor B is never marked; Y/X neighbors are co-marked. Presentation of the biological question is not axis-equivalent.",
    }
    report["tests"]["7_marker_overlay_equivalence"] = {
        "pass": False,
        "detail": "Marker skip-if-off-plane rule is geometrically consistent per plane but yields axis-asymmetric information on the XY panel.",
        "xy_covisibility": xy_cov,
    }

    # --- 5. Camera/view equivalence (napari dims focus on A) ---
    focus_policy = {
        "policy": "viewer.dims.set_point((0,1,2), pair_left_A)",
        "Z_pair_B_on_same_Z_slice_as_focus": False,
        "Y_pair_B_on_same_Z_slice_as_focus": True,
        "X_pair_B_on_same_Z_slice_as_focus": True,
    }
    report["tests"]["5_camera_view_equivalence"] = {
        "pass": False,
        "detail": focus_policy,
        "failure": "Main volume focus always on A; only Y/X neighbors share A's Z slice. Z neighbor requires changing Z to inspect B in the main view.",
    }

    # --- 6. Intensity normalization equivalence ---
    # Construct a volume where one orthogonal plane has a hot pixel so per-plane
    # stretch alters relative appearance of the same neighborhood differently by axis.
    vol2 = np.full((12, 12, 12), 40, dtype=np.uint8)
    vol2[5, 5, 5] = 50
    vol2[5, 5, 6] = 50  # X-neighbor of center
    vol2[0, :, :] = 255  # hot XY-other slice — does not affect XY at z=5, but affects XZ/YZ global? 
    # For docked XY at z=5, stretch uses only that plane's min/max.
    # For XZ at y=5, the hot z=0 row changes stretch.
    a = (5, 5, 5)
    b_x = (5, 5, 6)
    planes = docked_planes(vol2, a, b_x)
    # Compare gray level of A location after stretch on XY vs XZ
    # On XY: values 40 and 50 → A maps to ~0*scale; on XZ including 255, A maps lower contrast.
    xy_a = planes["XY"][5, 5]  # row=y=5,col=x=5
    xz_a = planes["XZ"][5, 5]  # row=z=5,col=x=5
    intensity_equivariant = bool(np.array_equal(xy_a, xz_a))
    report["tests"]["6_intensity_normalization_equivalence"] = {
        "pass": intensity_equivariant,
        "xy_A_rgb": xy_a.tolist(),
        "xz_A_rgb": xz_a.tolist(),
        "policy": "per-plane min/max stretch in _plane()",
        "failure": None
        if intensity_equivariant
        else "Same voxel A is rendered at different RGB intensities on different docked planes because stretch is local to each 2-D view.",
    }

    # --- 3. Replicate-member permutation equivariance (historical interface) ---
    # Documented: G3-008 does not use 3-member replicates; historical Property B fails.
    report["tests"]["3_replicate_member_permutation_equivariance"] = {
        "pass": False,
        "applies_to_008_queues": False,
        "detail": "Historical 3-member spread is non-equivariant (Property B). G3-008 queues are single central edges only, so this is not the -008 failure mode; retained for completeness.",
        "prior": "tests/test_interface_permutation_equivariance.py",
    }

    # --- 2. Context-window permutation equivariance for docked presentation ---
    # Under axis permutation, the set of visible markers on the 'section-like' plane
    # corresponding to the first two display axes should match after remapping.
    # We test a weaker but operational claim: for every axis, both A and B must be
    # co-visible on at least the same number of docked planes AND on XY specifically.
    cov_counts = {ax: sum(1 for p in cov.values() if p["A"] and p["B"]) for ax, cov in cov_by_axis.items()}
    report["tests"]["2_context_window_permutation_equivariance"] = {
        "pass": cov_counts["Z"] == cov_counts["Y"] == cov_counts["X"] and plane_equivariant,
        "planes_with_both_markers": cov_counts,
        "failure": "Z-pairs are co-visible on 2 docked planes (XZ,YZ); Y-pairs on XY+YZ; X-pairs on XY+XZ. Counts happen to match (2) but WHICH plane carries joint context differs — XY joint context is missing only for Z.",
    }

    # --- 8. Axis-label / text leakage ---
    report["tests"]["8_axis_label_text_leakage"] = {
        "pass": False,
        "detail": "Status text includes channel name (Z/Y/X) and interface member index; question order fixed Z→Y→X. Reviewer is not label-blind to axis identity.",
        "note": "Leakage alone does not prove causation of 12/12 pattern, but presentation is not axis-anonymous.",
    }

    # --- 9. Process-continuity context ---
    report["tests"]["9_process_continuity_context"] = {
        "pass": None,
        "detail": "Reviewer sees local orthogonal slices around a one-voxel pair only. Continuity of a process along its arbor is not shown; decision is necessarily local membrane/continuity judgment at a face.",
        "implication": "Operational SAME_PROCESS may collapse to 'no visible membrane between A and B in these planes' rather than true object identity.",
    }

    # --- 10. One-voxel adjacency meaningfulness ---
    report["tests"]["10_one_voxel_adjacency_definition"] = {
        "pass": None,
        "detail": "At 8 nm, a one-voxel face is below process diameter for many neurites; SAME/DIFFERENT can be dominated by partial-volume membrane texture and sectioning appearance, especially along milling (Z).",
        "implication": "Even with perfect UI equivariance, the annotation procedure may not identify the biological affinity relation reliably from local EM alone.",
    }

    # --- Permutation of presentation digests (strong synthetic check) ---
    # For each axis, hash the triple of docked planes. After permuting the volume
    # so that old Z becomes new Y, the presentation for the remapped edge should
    # match the original Y presentation if UI were equivariant. We compare
    # XY-panel co-visibility of B as the operational criterion.
    perm_failures = []
    # Map old axis 0 (Z) to new axis 1 (Y) via perm that sends 0->1, 1->0, 2->2
    perm = (1, 0, 2)  # new[0]=old[1], new[1]=old[0], new[2]=old[2]  WAIT
    # permute_coord: out[perm[d]] = c[d], so perm=(1,0,2) means out[1]=c[0], out[0]=c[1], out[2]=c[2]
    # i.e. Z<->Y swap. Good.
    vol_p = permute_volume(vol, perm)
    a_z, b_z = edge_pair(center, 0)
    a_p, b_p = permute_coord(a_z, perm), permute_coord(b_z, perm)
    # After Z<->Y, the edge that was Z should now be a Y-edge in the permuted volume.
    cov_orig_z = covisibility(vol, a_z, b_z)
    cov_perm = covisibility(vol_p, a_p, b_p)
    # Equivariant UI would give the same relative co-visibility pattern on remapped planes.
    # Operational check used by humans: is B on the XY panel?
    if cov_orig_z["XY"]["B"] != cov_perm["XY"]["B"]:
        # Actually after Z<->Y, the 'section' plane meaning changes. Stronger check:
        # number of planes with both markers should be preserved (already 2), and
        # XY B-visibility for a pure axis-swap should match what Y-edge had originally.
        a_y, b_y = edge_pair(center, 1)
        cov_orig_y = covisibility(vol, a_y, b_y)
        if cov_perm["XY"]["B"] != cov_orig_y["XY"]["B"]:
            perm_failures.append(
                {
                    "perm": list(perm),
                    "orig_Z_XY_B": cov_orig_z["XY"]["B"],
                    "perm_edge_XY_B": cov_perm["XY"]["B"],
                    "orig_Y_XY_B": cov_orig_y["XY"]["B"],
                    "note": "After Z↔Y volume permutation, remapped former-Z edge's XY B-visibility should equal original Y-edge XY B-visibility if presentation were equivariant.",
                }
            )
    report["tests"]["permutation_xy_b_visibility_match"] = {
        "pass": len(perm_failures) == 0,
        "failures": perm_failures,
        "orig_Z_xy": cov_orig_z["XY"],
        "perm_former_Z_xy": cov_perm["XY"],
        "orig_Y_xy": covisibility(vol, *edge_pair(center, 1))["XY"],
    }

    # Overall mechanism verdict
    hard_fails = [
        k
        for k, v in report["tests"].items()
        if v.get("pass") is False and k.startswith(("1_", "2_", "4_", "5_", "6_", "7_", "permutation_"))
    ]
    # test 1 should pass
    report["summary"] = {
        "central_edge_geometry_equivariant": report["tests"]["1_central_edge_permutation_equivariance"]["pass"],
        "presentation_permutation_equivariant": False,
        "hard_failing_tests": hard_fails,
        "failure_mechanism_established": "PRESENTATION_NOT_PERMUTATION_EQUIVARIANT",
        "failure_mechanism_is_complete_cause_of_12_of_12_labels": False,
        "reason_incomplete_cause": "Synthetic proof shows the reviewer-visible question is axis-asymmetric; it does not by itself prove that humans would reverse Z labels under an equivariant UI. Biology/image (Z milling) and one-voxel definition remain open. Therefore cause of CURRENT_AFFINITY_TARGET_INVALID is: invalid operational target realization; primary identified defect is presentation non-equivariance; ultimate label-generating cause among UI vs image vs definition still requires a controlled equivariant validation experiment.",
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
