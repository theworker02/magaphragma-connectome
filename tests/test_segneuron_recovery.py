"""Regression coverage for diagnostics that precede SegNeuron postprocessing."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from mvconnectome.segneuron_recovery import analyze_raw_prediction


class SegNeuronRecoveryTests(unittest.TestCase):
    def test_saturated_raw_prediction_is_recorded_before_postprocessing(self):
        with TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"; run.mkdir()
            (run / "inference.json").write_text(json.dumps({"input": {"sha256": "a" * 64}}), encoding="utf-8")
            np.save(run / "affinities.npy", np.ones((3, 2, 2, 2), dtype=np.float32))
            (run / "boundaries.tif").write_bytes(b"test-double")
            # TIFF is optional in the primary environment; use the same fake
            # extension only after supplying a lightweight test double.
            import sys, types
            previous = sys.modules.get("tifffile")
            sys.modules["tifffile"] = types.SimpleNamespace(imread=lambda _: np.ones((2, 2, 2), dtype=np.float32))
            try:
                result = analyze_raw_prediction(run, run / "diagnostic.json")
            finally:
                if previous is None:
                    del sys.modules["tifffile"]
                else:
                    sys.modules["tifffile"] = previous
            self.assertTrue(result["collapse_signals"]["network_output_near_saturated"])
            self.assertEqual(result["classification"], "MACHINE_PSEUDOLABEL_DIAGNOSTIC_ONLY")
