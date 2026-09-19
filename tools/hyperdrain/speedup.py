"""Speedup validation milestone — wall-clock per equivalent chunk, not vibes."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from hyperdrain.config import (
    BASELINE_COMPILE_TILES_PER_SEC,
    BASELINE_EAGER_TILES_PER_SEC,
    OUT,
    QUALIFIED_MANIFEST,
    ensure_out,
)
from hyperdrain.equivalence import compare_volumes, performance_receipt
from hyperdrain.geometry import CROP_ZYX, STRIDE_ZYX, redundancy_audit
from hyperdrain import backends, pipeline

SPEEDUP_VALIDATED_PATH = OUT / "HYPERDRAIN_SPEEDUP_VALIDATED.json"
DENSE_EXPERIMENTAL_PATH = OUT / "HYPERDRAIN_DENSE_EXPERIMENTAL.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def milestone_policy() -> dict:
    return {
        "id": "HYPERDRAIN_SPEEDUP_VALIDATED",
        "rule": (
            "May only be declared when an equivalence-passing backend demonstrates "
            "higher measured throughput than the frozen production baseline."
        ),
        "decisive_metric": "wall_seconds_per_equivalent_chunk",
        "secondary_metrics": [
            "tiles_per_second",
            "infer_seconds",
            "effective_output_voxels_per_second",
        ],
        "frozen_production_baseline": {
            "backend": "eager",
            "mode": "tiled",
            "crop_zyx": list(CROP_ZYX),
            "stride_zyx": list(STRIDE_ZYX),
            "reference_tiles_per_sec_constant": BASELINE_EAGER_TILES_PER_SEC,
            "compile_tiles_per_sec_constant_not_production": BASELINE_COMPILE_TILES_PER_SEC,
            "note": (
                "Live baseline wall_s is measured in the same validate-dense run "
                "(eager tiled on the same volume) AFTER a shared GPU warmup. "
                "Preserved tiles/s constants are not overwritten."
            ),
        },
        "dense_status_until_validated": "experimental",
        "do_not_call_faster_until": [
            "equivalence.pass == true",
            "wall_s_candidate < wall_s_baseline",
            "macro_zyx != production crop (real dense geometry)",
            "measurements taken after shared warmup (no cold-start artifact)",
        ],
    }


def _vox_per_sec(volume_zyx: tuple[int, int, int], wall_s: float) -> float | None:
    if wall_s <= 0:
        return None
    return float(volume_zyx[0] * volume_zyx[1] * volume_zyx[2]) / wall_s


def _warmup(model: Any, volume: np.ndarray, batch_size: int) -> None:
    """Full discard pass on the validation volume so timed legs are not cold-start."""
    import torch

    print("validate-dense: discard warmup pass (production geometry, untimed)...", flush=True)
    pipeline.infer_volume_tiled(
        model,
        volume,
        batch_size=max(1, int(batch_size)),
        crop_zyx=CROP_ZYX,
        stride_zyx=STRIDE_ZYX,
        use_amp=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _sync_production_manifest(*, speedup_validated: bool, dense_note: str) -> None:
    q = backends.load_qualified()
    q["speedup_validated"] = bool(speedup_validated)
    q["decisive_metric"] = "wall_seconds_per_equivalent_chunk"
    experimental = list(q.get("experimental") or [])
    for name in ("migraphx", "compile", "dense"):
        if name not in experimental:
            experimental.append(name)
    if not speedup_validated and "dense" not in experimental:
        experimental.append("dense")
    if speedup_validated and "dense" in experimental:
        # Validated dense may leave experimental list but must also be qualified.
        pass
    q["experimental"] = experimental
    if speedup_validated:
        qualified = list(q.get("qualified") or [])
        if "dense" not in qualified:
            qualified.append("dense")
        q["qualified"] = qualified
    notes = dict(q.get("notes") or {})
    notes["dense"] = dense_note
    q["notes"] = notes
    backends.save_qualified(q)


def run_dense_speedup_validation(
    model_baseline: Any,
    model_candidate: Any,
    volume: np.ndarray,
    *,
    macro_zyx: tuple[int, int, int],
    baseline_batch: int = 16,
    abs_tol: float = 1e-3,
) -> dict:
    """
    Compare eager-tiled baseline vs dense/macro candidate on the SAME volume.

    Decisive: wall_seconds for a full equivalent infer+stitch of that volume.
    SPEEDUP_VALIDATED written only if:
      - equivalence PASS
      - candidate wall_s is strictly lower
      - macro geometry differs from frozen production crop (otherwise not dense)
      - both legs timed after a shared warmup
    """
    ensure_out()
    policy = milestone_policy()
    audit = redundancy_audit(tuple(volume.shape))
    macro_zyx = tuple(int(v) for v in macro_zyx)
    is_real_dense = macro_zyx != tuple(CROP_ZYX)

    print(
        f"validate-dense warmup (shared) before timed legs; macro_zyx={macro_zyx} "
        f"real_dense={is_real_dense}",
        flush=True,
    )
    _warmup(model_baseline, volume, baseline_batch)

    # --- baseline: frozen production geometry (warm) ---
    t0 = time.perf_counter()
    ref_aff, ref_bnd, ref_tel = pipeline.infer_volume_tiled(
        model_baseline,
        volume,
        batch_size=baseline_batch,
        crop_zyx=CROP_ZYX,
        stride_zyx=STRIDE_ZYX,
        use_amp=True,
    )
    baseline_wall = time.perf_counter() - t0
    ref_tel = {
        **ref_tel,
        "wall_seconds_per_equivalent_chunk": baseline_wall,
        "effective_output_voxels_per_second": _vox_per_sec(tuple(volume.shape), baseline_wall),
        "role": "frozen_production_baseline",
    }

    # --- candidate: dense/macro (experimental until validated) ---
    t1 = time.perf_counter()
    cand_aff, cand_bnd, cand_tel = pipeline.infer_volume_macro(
        model_candidate,
        volume,
        macro_zyx=macro_zyx,
        use_amp=True,
    )
    cand_wall = time.perf_counter() - t1
    cand_tel = {
        **cand_tel,
        "wall_seconds_per_equivalent_chunk": cand_wall,
        "effective_output_voxels_per_second": _vox_per_sec(tuple(volume.shape), cand_wall),
        "macro_zyx": list(macro_zyx),
        "role": "dense_experimental_candidate",
    }

    cmp = compare_volumes(ref_aff, cand_aff, ref_bnd, cand_bnd, abs_tol=abs_tol)
    wall_ratio = (baseline_wall / cand_wall) if cand_wall > 0 else None
    wall_improved = bool(cmp.get("pass") and cand_wall < baseline_wall and baseline_wall > 0)
    # Same crop as production is a harness control, not a dense acceleration claim.
    declare = bool(wall_improved and is_real_dense)

    if declare:
        verdict = "HYPERDRAIN_SPEEDUP_VALIDATED"
    elif not is_real_dense and wall_improved and cmp.get("pass"):
        verdict = "HARNESS_CONTROL_SAME_GEOMETRY_NOT_DENSE"
    elif not cmp.get("pass"):
        verdict = "EXPERIMENTAL_EQUIVALENCE_FAIL"
    else:
        verdict = "EXPERIMENTAL_NO_WALL_IMPROVEMENT"

    result = {
        "id": "HYPERDRAIN_DENSE_SPEEDUP_VALIDATION",
        "created_at": _now(),
        "policy": policy,
        "volume_zyx": list(volume.shape),
        "redundancy_before": audit.get("redundancy_before"),
        "macro_zyx": list(macro_zyx),
        "is_real_dense_geometry": is_real_dense,
        "warmup": "full_production_geometry_discard_pass_before_timed_legs",
        "baseline": ref_tel,
        "candidate": cand_tel,
        "equivalence": cmp,
        "wall_seconds_baseline": baseline_wall,
        "wall_seconds_candidate": cand_wall,
        "wall_speedup_factor": wall_ratio if cmp.get("pass") else None,
        "wall_ratio_informational": wall_ratio,
        "tiles_per_sec_baseline": ref_tel.get("tiles_per_second"),
        "tiles_per_sec_candidate": cand_tel.get("tiles_per_second"),
        "SPEEDUP_VALIDATED": declare,
        "dense_label": "validated_acceleration" if declare else "experimental",
        "verdict": verdict,
    }

    DENSE_EXPERIMENTAL_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if declare:
        validated = {
            "id": "HYPERDRAIN_SPEEDUP_VALIDATED",
            "declared_at": _now(),
            "backend": "dense",
            "macro_zyx": list(macro_zyx),
            "wall_seconds_baseline": baseline_wall,
            "wall_seconds_candidate": cand_wall,
            "wall_speedup_factor": wall_ratio,
            "equivalence_pass": True,
            "decisive_metric": "wall_seconds_per_equivalent_chunk",
            "evidence": str(DENSE_EXPERIMENTAL_PATH),
        }
        SPEEDUP_VALIDATED_PATH.write_text(
            json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        dense_note = (
            f"VALIDATED {wall_ratio:.3f}x wall vs eager tiled on {list(volume.shape)} "
            f"with macro {list(macro_zyx)}; decisive metric wall_seconds_per_equivalent_chunk."
        )
    else:
        if SPEEDUP_VALIDATED_PATH.exists():
            SPEEDUP_VALIDATED_PATH.unlink()
        dense_note = (
            "Experimental until HYPERDRAIN_SPEEDUP_VALIDATED: equivalence PASS and "
            "wall_seconds_per_equivalent_chunk strictly below measured eager-tiled baseline "
            f"on a real dense macro (not production crop). Last verdict={verdict}. "
            "tiles/s alone is not sufficient."
        )

    _sync_production_manifest(speedup_validated=declare, dense_note=dense_note)

    performance_receipt(
        measurements=[
            {
                "role": "baseline_eager_tiled",
                "wall_seconds_per_equivalent_chunk": baseline_wall,
                "tiles_per_second": ref_tel.get("tiles_per_second"),
            },
            {
                "role": "candidate_dense_macro",
                "wall_seconds_per_equivalent_chunk": cand_wall,
                "tiles_per_second": cand_tel.get("tiles_per_second"),
                "macro_zyx": list(macro_zyx),
                "equivalence_pass": cmp.get("pass"),
            },
        ],
        notes=(
            "Dense remains experimental unless SPEEDUP_VALIDATED. "
            "Decisive metric is wall_seconds_per_equivalent_chunk. "
            f"verdict={verdict}"
        ),
    )
    return result
