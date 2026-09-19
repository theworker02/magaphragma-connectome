"""Unit tests for label-backed instance metrics and error taxonomy."""

import unittest

import numpy as np

from mvconnectome.qualification import instance_metrics


class QualificationTests(unittest.TestCase):
    def test_instance_metrics_expose_merge_split_miss_and_false_objects(self):
        truth = np.array([[[1, 1, 2, 2, 3, 3]]], dtype=np.uint32)
        prediction = np.array([[[7, 7, 7, 7, 0, 8]]], dtype=np.uint32)
        metrics = instance_metrics(truth, prediction)
        self.assertEqual(metrics["merge_errors"], 1)
        self.assertEqual(metrics["split_errors"], 0)
        self.assertEqual(metrics["missed_objects"], 0)
        self.assertEqual(metrics["false_objects"], 0)
        self.assertGreater(metrics["vi_total"], 0)
