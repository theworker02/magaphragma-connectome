"""Regression coverage for G2 diagnostic queues, not biological decisions."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mvconnectome.g2_diagnostics import create_g2_diagnostic_queue
from mvconnectome.io import sha256_file


def test_g2_queue_requires_frozen_region_and_checkpoint() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary); raw = root / "raw.npy"
        np.save(raw, np.arange(4 * 6 * 6, dtype=np.uint8).reshape(4, 6, 6))
        raw_hash, checkpoint_hash = sha256_file(raw), "a" * 64
        cohort = root / "cohort.json"
        cohort.write_text(json.dumps({"status": "FROZEN_BEFORE_G1_DIAGNOSTIC_REVIEW", "regions": [{"id": "R", "role": "G2_TARGET_TRAIN", "raw_path": str(raw), "raw_sha256": raw_hash, "shape_zyx": [4, 6, 6]}]}))
        inference = root / "inference"; inference.mkdir(); np.save(inference / "aff.npy", np.full((3, 4, 6, 6), 0.95, dtype=np.float32))
        (inference / "inference.json").write_text(json.dumps({"input": {"sha256": raw_hash, "shape_zyx": [4, 6, 6]}, "checkpoint": {"sha256": checkpoint_hash}, "affinities": {"file": "aff.npy"}}))
        queue = create_g2_diagnostic_queue(cohort_path=cohort, region_id="R", inference_dir=inference, output_path=root / "queue.json", expected_checkpoint_sha256=checkpoint_hash, candidate_count=1, control_count=1, minimum_distance=1)
        assert queue["status"] == "EXPERT_REVIEW_REQUIRED"
        assert all(question["g1_affinity"] >= 0.9 for question in queue["questions"])
