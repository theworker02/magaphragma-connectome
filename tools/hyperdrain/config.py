"""Shared paths and frozen scientific constants for HyperDrain."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"
OUT = REPO / "experiments/phase6e/HYPERDRAIN"
S7_OUT = REPO / "experiments/phase6e/AFFINITY-FULLVOL-S7-001"
CHUNKS = REPO / "local_research_build/phase5c-production/chunks.json"
ROI_PKG = REPO / "experiments/phase6e/AFFINITY-ROI-001/package/ROI_PACKAGE.json"
S6 = REPO / "experiments/phase6e/AFFINITY_S6_SEG_QC_AUTHORIZATION_001.json"
CKPT = REPO / "experiments/phase6e/AFFINITY-PARALLEL-FUNNEL-001/C_PATCH_20_64_64/extend/checkpoints/checkpoint-step10.pt"
DVID_BASE = (
    "https://waspem-dvid2.flatironinstitute.org/api/node/"
    "aa49d16e88424cdbae89f4b8ced2a49b/five_yuri_4contrast/raw/0_1_2/"
)

# Preserved measured baselines — never overwrite; only compare against.
BASELINE_EAGER_TILES_PER_SEC = 18.9
BASELINE_COMPILE_TILES_PER_SEC = 20.8

VRAM_CACHE = OUT / "vram_config_cache.json"
QUALIFIED_MANIFEST = OUT / "HYPERDRAIN_PRODUCTION_BACKEND.json"
EQUIV_RECEIPT = OUT / "HYPERDRAIN_EQUIVALENCE_RECEIPT.json"
PERF_RECEIPT = OUT / "HYPERDRAIN_PERFORMANCE_RECEIPT.json"
QUEUE_STATE = OUT / "queue_state.json"
AUDIT_PATH = OUT / "REDUNDANCY_AUDIT.json"

SEGNEURON_TRAIN = REPO / "third_party/segneuron/Train_and_Inference"


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def env_backend(default: str = "auto") -> str:
    return (os.environ.get("HYPERDRAIN_BACKEND") or default).strip().lower()
