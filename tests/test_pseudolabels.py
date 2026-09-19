"""Regression coverage for explicit non-ground-truth proposal generation."""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import json

import numpy as np

from mvconnectome.pseudolabels import audit_existing_proposal, generate_interior_component_proposals


class PseudolabelTests(unittest.TestCase):
    def test_proposals_are_explicitly_non_ground_truth_and_qa_flagged(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); run = root / "run"; run.mkdir()
            (run / "inference.json").write_text(json.dumps({"checkpoint":{"sha256":"a" * 64},"model":"test"}), encoding="utf-8")
            np.save(run / "affinities.npy", np.full((3, 3, 3, 3), 0.5, dtype=np.float32))
            receipt = root / "receipt.json"; receipt.write_text("{}", encoding="utf-8")
            result = generate_interior_component_proposals(run, receipt, root / "proposal.npy", root / "qa.json", minimum_voxels=1)
            self.assertEqual(result["status"], "MACHINE_PSEUDOLABEL")
            self.assertGreater(result["proposal_instances"], 0)

    def test_audit_flags_near_full_volume_component(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); proposal = root / "proposal.npy"
            np.save(proposal, np.ones((3, 3, 3), dtype=np.uint32))
            result = audit_existing_proposal(proposal, root / "audit.json")
            self.assertEqual(result["near_full_volume_merge_candidates"], 1)
