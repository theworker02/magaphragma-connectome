"""Synthetic absolute gate: presentation equivariance under all 6 axis permutations.

Fail-closed. Writes experiments/phase6e/G3_PRESENTATION_EQUIVARIANCE_REPAIR_001.json
Exit code 0 only if every fixture × every permutation passes.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from mvconnectome.equivariant_edge_presentation import (  # noqa: E402
    both_markers_visible,
    presentation_fingerprint,
    render_edge_presentation,
)

OUT = REPO / "experiments/phase6e/G3_PRESENTATION_EQUIVARIANCE_REPAIR_001.json"
PERMS = list(itertools.permutations((0, 1, 2)))
RADIUS = 8


def permute_coord(c: tuple[int, int, int], perm: tuple[int, int, int]) -> tuple[int, int, int]:
    out = [0, 0, 0]
    for d in range(3):
        out[perm[d]] = c[d]
    return tuple(out)


def permute_volume(vol: np.ndarray, perm: tuple[int, int, int]) -> np.ndarray:
    """Volume transform consistent with permute_coord: vol_p[π(c)] = vol[c]."""
    inv = [0, 0, 0]
    for d in range(3):
        inv[perm[d]] = d
    return np.transpose(vol, axes=tuple(inv))


def fixtures() -> dict[str, tuple[np.ndarray, tuple[int, int, int], int, int]]:
    """name -> (volume, pair_left, channel, expected_affinity 1/0)."""
    n = 24
    mid = n // 2
    out = {}

    # uniform same-object
    vol = np.full((n, n, n), 120, dtype=np.uint8)
    out["uniform_same"] = (vol, (mid, mid, mid), 0, 1)

    # planar membrane separating objects (normal Z)
    vol = np.full((n, n, n), 40, dtype=np.uint8)
    vol[: mid + 1] = 40
    vol[mid + 1 :] = 200
    vol[mid] = 255
    out["planar_membrane_Z"] = (vol, (mid, mid, mid), 0, 0)  # crosses membrane

    # same planar, but Y-edge stays in one object
    out["planar_membrane_Y_same"] = (vol.copy(), (mid - 2, mid, mid), 1, 1)

    # diagonal / oblique boundary: plane z+y = const
    zz, yy, xx = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    vol = np.where(zz + yy < n, 50, 180).astype(np.uint8)
    vol = np.where(np.abs(zz + yy - (n - 1)) <= 0, 255, vol)
    out["oblique_boundary"] = (vol, (mid - 1, mid, mid), 0, 0)

    # thin process along X
    vol = np.full((n, n, n), 20, dtype=np.uint8)
    vol[mid - 1 : mid + 2, mid - 1 : mid + 2, :] = 160
    out["thin_process_X"] = (vol, (mid, mid, mid), 2, 1)

    # curved / tubular process (circle in YX, extruded in Z)
    vol = np.full((n, n, n), 15, dtype=np.uint8)
    cy, cx, r = mid, mid, 5
    for y in range(n):
        for x in range(n):
            d = abs(((y - cy) ** 2 + (x - cx) ** 2) ** 0.5 - r)
            if d < 1.5:
                vol[:, y, x] = 170
    out["tubular_process"] = (vol, (mid, mid + r, mid), 0, 1)

    # edge near membrane (Z-edge just before sheet)
    vol = np.full((n, n, n), 60, dtype=np.uint8)
    vol[mid + 1] = 255
    vol[mid + 2 :] = 90
    out["edge_near_membrane"] = (vol, (mid, mid, mid), 0, 0)

    # symmetric tied geometry: cross of two bright bars
    vol = np.full((n, n, n), 30, dtype=np.uint8)
    vol[mid - 1 : mid + 2, :, mid - 1 : mid + 2] = 150
    vol[:, mid - 1 : mid + 2, mid - 1 : mid + 2] = 150
    out["symmetric_cross"] = (vol, (mid, mid, mid), 1, 1)

    return out


def check_presentation_invariants(pres, errors: list) -> None:
    vis = both_markers_visible(pres)
    # Side A must show A; Side B must show B; both longitudinal must show A and B
    if not vis["SIDE_A"]["A"]:
        errors.append("SIDE_A missing A")
    if not vis["SIDE_B"]["B"]:
        errors.append("SIDE_B missing B")
    for name in ("LONGITUDINAL_1", "LONGITUDINAL_2"):
        if not (vis[name]["A"] and vis[name]["B"]):
            errors.append(f"{name} missing co-visible A/B: {vis[name]}")
    if pres.fov_nm != (2 * RADIUS + 1) * 8.0:
        errors.append(f"fov_nm mismatch {pres.fov_nm}")
    if len(pres.panel_order) != 4:
        errors.append("expected 4 panels")
    shapes = {pres.panels[n].shape for n in pres.panel_order}
    if len(shapes) != 1:
        errors.append(f"panel shape mismatch {shapes}")


def main() -> int:
    report = {
        "id": "MV-G3-PRESENTATION-EQUIVARIANCE-REPAIR-001",
        "schema_version": 1,
        "status": "PENDING",
        "target_spec": "MV-G3-AFFINITY-TARGET-SPEC-002",
        "radius_voxels": RADIUS,
        "voxel_nm": 8.0,
        "panel_contract": ["SIDE_A", "SIDE_B", "LONGITUDINAL_1", "LONGITUDINAL_2"],
        "fixtures": {},
        "failures": [],
    }
    all_ok = True

    for name, (vol, left, channel, _aff) in fixtures().items():
        fx = {"channel_zyx": channel, "pair_left_zyx": list(left), "permutation_results": [], "pass": True}
        try:
            base = render_edge_presentation(vol, left, channel, radius=RADIUS)
        except Exception as exc:  # noqa: BLE001
            fx["pass"] = False
            fx["error"] = str(exc)
            all_ok = False
            report["failures"].append({"fixture": name, "error": str(exc)})
            report["fixtures"][name] = fx
            continue

        inv_err: list[str] = []
        check_presentation_invariants(base, inv_err)
        base_fp = presentation_fingerprint(base)
        fx["base_fingerprint"] = base_fp
        fx["base_visibility"] = both_markers_visible(base)
        if inv_err:
            fx["pass"] = False
            all_ok = False
            report["failures"].append({"fixture": name, "invariant": inv_err})

        for perm in PERMS:
            vol_p = permute_volume(vol, perm)
            left_p = permute_coord(tuple(left), perm)
            ch_p = perm[channel]
            try:
                pres_p = render_edge_presentation(vol_p, left_p, ch_p, radius=RADIUS)
            except Exception as exc:  # noqa: BLE001
                fx["pass"] = False
                all_ok = False
                report["failures"].append({"fixture": name, "perm": list(perm), "error": str(exc)})
                continue
            err: list[str] = []
            check_presentation_invariants(pres_p, err)
            fp = presentation_fingerprint(pres_p)
            # Equivariance: after remapping into the local edge frame, panel
            # fingerprints must match the unpermuted presentation.
            if fp != base_fp:
                err.append("panel_fingerprint_mismatch_vs_base")
            entry = {"perm": list(perm), "channel_zyx": ch_p, "fingerprint": fp, "errors": err, "pass": not err}
            fx["permutation_results"].append(entry)
            if err:
                fx["pass"] = False
                all_ok = False
                report["failures"].append({"fixture": name, "perm": list(perm), "errors": err})

        report["fixtures"][name] = fx

    # Additional: no axis text in panel contract names (structural)
    report["axis_text_absent_from_panel_names"] = all(
        ax not in " ".join(report["panel_contract"]) for ax in ("Z", "Y", "X")
    )
    if not report["axis_text_absent_from_panel_names"]:
        all_ok = False
        report["failures"].append({"error": "panel names contain axis letters"})

    report["overall_pass"] = all_ok
    report["status"] = "PASS_EQUIVARIANT" if all_ok else "FAIL_CLOSED"
    report["decision"] = (
        "SYNTHETIC_GATE_PASSED_HUMAN_SMOKE_MAY_PROCEED"
        if all_ok
        else "SYNTHETIC_GATE_FAILED_DO_NOT_COLLECT_HUMAN_LABELS"
    )
    digest = hashlib.sha256(json.dumps(report["fixtures"], sort_keys=True, default=str).encode()).hexdigest()
    report["fixtures_sha256"] = digest

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "overall_pass": all_ok, "failures": len(report["failures"])}, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
