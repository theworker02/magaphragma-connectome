"""Regression coverage for independent raw-EM boundary-review evidence."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mvconnectome.external_boundary_review import (
    append_external_boundary_event,
    create_external_scan_queue,
    create_external_review_workspace,
    materialize_external_boundary_affinity,
)
from mvconnectome.io import sha256_file


def _fixture(root: Path) -> tuple[Path, Path]:
    raw = root / "raw.npy"
    np.save(raw, np.zeros((3, 4, 5), dtype=np.uint8))
    digest = sha256_file(raw)
    request = root / "request.json"
    request.write_text(json.dumps({"id": "request", "requested_review_sets": [{
        "crop_id": "MV-DVID-ANN-000001", "role": "DVID_TARGET_TRAIN", "parent_region_id": "MV-GTVOL-000001",
        "raw_sha256": digest, "coordinate_frame": "frame", "source_origin_xyz": [0, 0, 0],
    }]}), encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"crops": [{
        "id": "MV-DVID-ANN-000001", "parent_region_id": "MV-GTVOL-000001", "raw_crop_path": str(raw),
        "raw_crop_sha256": digest, "raw_shape_zyx": [3, 4, 5], "raw_dtype": "uint8",
        "coordinate_frame": "frame", "source_origin_xyz": [0, 0, 0],
    }]}), encoding="utf-8")
    return request, manifest


def test_external_decisions_materialize_masked_affinities() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary); request, manifest = _fixture(root); workspace = root / "workspace.json"
        create_external_review_workspace(request_path=request, crop_manifest_path=manifest, crop_id="MV-DVID-ANN-000001", output=workspace)
        append_external_boundary_event(workspace, reviewer="expert", decision="SAME_PROCESS", pair_left_zyx=[1, 1, 1], channel_zyx=2, orthogonal_views_inspected=True)
        append_external_boundary_event(workspace, reviewer="expert", decision="DIFFERENT_PROCESS", pair_left_zyx=[1, 2, 2], channel_zyx=1, orthogonal_views_inspected=True)
        receipt = materialize_external_boundary_affinity(workspace, root / "targets")
        assert receipt["counts"]["SAME_PROCESS"] == 1
        assert receipt["counts"]["DIFFERENT_PROCESS"] == 1
        assert receipt["counts"]["mask_pairs"] == 2


def test_external_pair_requires_orthogonal_review_and_in_bounds_geometry() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary); request, manifest = _fixture(root); workspace = root / "workspace.json"
        create_external_review_workspace(request_path=request, crop_manifest_path=manifest, crop_id="MV-DVID-ANN-000001", output=workspace)
        try:
            append_external_boundary_event(workspace, reviewer="expert", decision="SAME_PROCESS", pair_left_zyx=[1, 1, 1], channel_zyx=0)
        except ValueError as error:
            assert "XY, XZ, YZ" in str(error)
        else:
            raise AssertionError("expected orthogonal-view confirmation rejection")
        try:
            append_external_boundary_event(workspace, reviewer="expert", decision="SAME_PROCESS", pair_left_zyx=[2, 1, 1], channel_zyx=0, orthogonal_views_inspected=True)
        except ValueError as error:
            assert "outside" in str(error)
        else:
            raise AssertionError("expected out-of-bounds pair rejection")


def test_external_placeholder_reviewer_is_rejected() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary); request, manifest = _fixture(root); workspace = root / "workspace.json"
        create_external_review_workspace(request_path=request, crop_manifest_path=manifest, crop_id="MV-DVID-ANN-000001", output=workspace)
        try:
            append_external_boundary_event(workspace, reviewer="expert-identifier", decision="SAME_PROCESS", pair_left_zyx=[1, 1, 1], channel_zyx=2, orthogonal_views_inspected=True)
        except ValueError as error:
            assert "reviewer" in str(error).lower()
        else:
            raise AssertionError("expected placeholder reviewer rejection")


def test_qualified_reviewer_can_replace_placeholder_pair_without_erasing_history() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary); request, manifest = _fixture(root); workspace = root / "workspace.json"
        create_external_review_workspace(request_path=request, crop_manifest_path=manifest, crop_id="MV-DVID-ANN-000001", output=workspace)
        log = Path(json.loads(workspace.read_text(encoding="utf-8"))["event_log"]["path"])
        log.write_text(json.dumps({"pair_left_zyx": [1, 1, 1], "channel_zyx": 2, "reviewer": "expert-identifier", "decision": "UNCERTAIN"}) + "\n", encoding="utf-8")
        event = append_external_boundary_event(workspace, reviewer="matth", decision="BAD_QUESTION", pair_left_zyx=[1, 1, 1], channel_zyx=2, orthogonal_views_inspected=True)
        assert event["decision"] == "BAD_QUESTION"
        assert len(log.read_text(encoding="utf-8").splitlines()) == 2


def test_external_scan_queue_is_spatial_navigation_not_model_selection() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary); request, manifest = _fixture(root); workspace = root / "workspace.json"
        create_external_review_workspace(request_path=request, crop_manifest_path=manifest, crop_id="MV-DVID-ANN-000001", output=workspace)
        queue = create_external_scan_queue(workspace_path=workspace, output=root / "scan.json", count=8, border=0)
        assert queue["selection"]["method"] == "SPATIALLY_STRATIFIED_GRID_V1"
        assert len(queue["locations_zyx"]) == 8
        assert "SegNeuron" in queue["selection"]["prohibited_inputs"]


def test_external_event_preserves_diagnostic_question_reference() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary); request, manifest = _fixture(root); workspace = root / "workspace.json"
        create_external_review_workspace(request_path=request, crop_manifest_path=manifest, crop_id="MV-DVID-ANN-000001", output=workspace)
        event = append_external_boundary_event(workspace, reviewer="expert", decision="DIFFERENT_PROCESS", pair_left_zyx=[1, 1, 1], channel_zyx=2, orthogonal_views_inspected=True, question_reference="MERGE_CONFIDENT_RAW_BOUNDARY_CANDIDATE:1")
        assert event["question_reference"] == "MERGE_CONFIDENT_RAW_BOUNDARY_CANDIDATE:1"
