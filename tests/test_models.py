"""Domain-invariant tests that stop identifiers and review states drifting."""

import unittest

from mvconnectome.ids import stable_id
from mvconnectome.models import Evidence, EvidenceStatus, Neuron, ReviewState, Synapse


class ModelTests(unittest.TestCase):
    def test_machine_evidence_requires_model_hash(self) -> None:
        with self.assertRaises(ValueError):
            Evidence("MV-EV-000001", EvidenceStatus.MACHINE_PREDICTED, "MV-SRC-WASPSYN-2024", "MV-VOL-000001", (1, 2, 3), None, None, "detector")

    def test_neuron_requires_evidence_and_segments(self) -> None:
        with self.assertRaises(ValueError):
            Neuron("MV-N-000001", (), ("MV-SRC-WASPSYN-2024",), EvidenceStatus.UNKNOWN, ReviewState.UNREVIEWED, ())

    def test_verified_synapse_needs_human_review(self) -> None:
        with self.assertRaises(ValueError):
            Synapse("MV-SYN-00000001", "MV-EV-000001", "MV-N-000001", "MV-N-000002", ("MV-SEG-000001",), (1, 2, 3), EvidenceStatus.MANUALLY_VERIFIED, ReviewState.MACHINE_ONLY)

    def test_stable_ids_are_not_row_ids(self) -> None:
        self.assertEqual(stable_id("synapse", 14), "MV-SYN-00000014")
