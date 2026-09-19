"""Regression coverage for immutable, split-safe annotation-crop handling."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mvconnectome.annotation_crops import IGNORE_LABEL, materialize_annotation_crops, register_reviewed_crop_label


def _write_parent(root: Path) -> Path:
    raw = root / "parent.npy"
    np.save(raw, np.arange(8 * 8 * 8, dtype=np.uint8).reshape(8, 8, 8))
    manifest = root / "parents.json"
    manifest.write_text(json.dumps({"regions": [
        {"id": "MV-GTVOL-000001", "dimensions_voxels_xyz": [8, 8, 8], "raw_path": str(raw), "raw_sha256": "a" * 64, "bounds_xyz": {"x": [0, 8], "y": [0, 8], "z": [0, 8]}, "coordinate_frame": "frame", "dataset_id": "source", "volume_id": "volume"}
    ]}), encoding="utf-8")
    return manifest


def test_crops_materialize_and_reviewed_labels_preserve_ignore():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        parents = _write_parent(root)
        plan = root / "plan.json"
        plan.write_text(json.dumps({"id": "plan", "crops": [{"id": "MV-DVID-ANN-000001", "parent_region_id": "MV-GTVOL-000001", "split": "DVID_TARGET_TRAIN", "origin_zyx": [1, 1, 1], "shape_zyx": [4, 4, 4]}]}), encoding="utf-8")
        manifest = root / "crops.json"
        materialize_annotation_crops(plan, parents, root / "raw", manifest)
        label = root / "reviewed.npy"
        values = np.zeros((4, 4, 4), dtype=np.uint32); values[0, 0, 0] = 1; values[1, 1, 1] = IGNORE_LABEL
        np.save(label, values)
        receipt = register_reviewed_crop_label(manifest, "MV-DVID-ANN-000001", label, "reviewer", root / "receipts" / "label.json")
        assert receipt["status"] == "REVIEWED_DVID_LABEL"
        assert receipt["ignore_voxels"] == 1


def test_regression_parent_is_rejected():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        parents = _write_parent(root)
        plan = root / "plan.json"
        plan.write_text(json.dumps({"id": "plan", "crops": [{"id": "MV-DVID-ANN-000001", "parent_region_id": "MV-GTVOL-000004", "split": "DVID_TARGET_TRAIN", "origin_zyx": [0, 0, 0], "shape_zyx": [4, 4, 4]}]}), encoding="utf-8")
        try:
            materialize_annotation_crops(plan, parents, root / "raw", root / "crops.json")
        except ValueError as error:
            assert "Regression-only" in str(error)
        else:
            raise AssertionError("expected regression parent to be rejected")
