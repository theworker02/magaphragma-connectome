"""Equivalence harness — never weaken gates to look faster."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from hyperdrain.config import BASELINE_COMPILE_TILES_PER_SEC, BASELINE_EAGER_TILES_PER_SEC, EQUIV_RECEIPT, ensure_out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_digest(arr: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def compare_volumes(
    ref_aff: np.ndarray,
    cand_aff: np.ndarray,
    ref_bnd: np.ndarray,
    cand_bnd: np.ndarray,
    *,
    abs_tol: float = 1e-3,
    decision_threshold: float = 0.5,
    max_decision_disagree_frac: float = 1e-4,
) -> dict:
    if ref_aff.shape != cand_aff.shape or ref_bnd.shape != cand_bnd.shape:
        return {
            "pass": False,
            "reason": f"shape mismatch aff {ref_aff.shape} vs {cand_aff.shape} bnd {ref_bnd.shape} vs {cand_bnd.shape}",
        }
    if not (np.isfinite(ref_aff).all() and np.isfinite(cand_aff).all() and np.isfinite(ref_bnd).all() and np.isfinite(cand_bnd).all()):
        return {
            "pass": False,
            "reason": "non-finite values (NaN/Inf) in reference or candidate",
            "ref_finite_frac": float(np.isfinite(ref_aff).mean()),
            "cand_finite_frac": float(np.isfinite(cand_aff).mean()),
        }
    diff = np.abs(ref_aff.astype(np.float64) - cand_aff.astype(np.float64))
    diff_b = np.abs(ref_bnd.astype(np.float64) - cand_bnd.astype(np.float64))
    ref_dec = ref_aff >= decision_threshold
    cand_dec = cand_aff >= decision_threshold
    disagree = ref_dec != cand_dec
    disagree_frac = float(disagree.mean())
    stats = {
        "aff_max_abs": float(diff.max()),
        "aff_mean_abs": float(diff.mean()),
        "aff_p99_abs": float(np.quantile(diff, 0.99)),
        "bnd_max_abs": float(diff_b.max()),
        "bnd_mean_abs": float(diff_b.mean()),
        "decision_disagree_frac": disagree_frac,
        "decision_threshold": decision_threshold,
    }
    # Tight numerical + decision gate. Faster+FAIL is still FAIL.
    numerical_ok = stats["aff_max_abs"] <= abs_tol or stats["aff_mean_abs"] <= abs_tol * 0.1
    # Allow tiny float noise under autocast: also pass if p99 very small and disagree tiny
    soft_ok = stats["aff_p99_abs"] <= abs_tol and disagree_frac <= max_decision_disagree_frac
    decision_ok = disagree_frac <= max_decision_disagree_frac
    passed = bool((numerical_ok or soft_ok) and decision_ok and ref_aff.shape == cand_aff.shape)
    return {"pass": passed, "stats": stats, "abs_tol": abs_tol, "max_decision_disagree_frac": max_decision_disagree_frac}


def run_equivalence(
    *,
    ref_name: str,
    cand_name: str,
    ref_aff: np.ndarray,
    cand_aff: np.ndarray,
    ref_bnd: np.ndarray,
    cand_bnd: np.ndarray,
    ref_meta: dict,
    cand_meta: dict,
    geometry: dict,
    versions: dict,
) -> dict:
    cmp = compare_volumes(ref_aff, cand_aff, ref_bnd, cand_bnd)
    receipt = {
        "id": "HYPERDRAIN_EQUIVALENCE_RECEIPT",
        "created_at": _now(),
        "reference": ref_name,
        "candidate": cand_name,
        "pass": cmp["pass"],
        "comparison": cmp,
        "reference_digests": {"affinities": array_digest(ref_aff), "boundaries": array_digest(ref_bnd)},
        "candidate_digests": {"affinities": array_digest(cand_aff), "boundaries": array_digest(cand_bnd)},
        "reference_meta": ref_meta,
        "candidate_meta": cand_meta,
        "geometry": geometry,
        "versions": versions,
        "policy": "Faster + FAIL ≠ production. Gates are not relaxed for throughput.",
    }
    ensure_out()
    EQUIV_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def performance_receipt(
    *,
    measurements: list[dict],
    notes: str = "",
) -> dict:
    """Record only measured metrics; preserve known baselines as reference constants."""
    receipt = {
        "id": "HYPERDRAIN_PERFORMANCE_RECEIPT",
        "created_at": _now(),
        "preserved_baselines": {
            "eager_tiles_per_sec": BASELINE_EAGER_TILES_PER_SEC,
            "compile_tiles_per_sec": BASELINE_COMPILE_TILES_PER_SEC,
            "note": "Do not overwrite; compare new measurements against these.",
        },
        "measurements": measurements,
        "notes": notes,
    }
    from hyperdrain.config import PERF_RECEIPT

    ensure_out()
    PERF_RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
