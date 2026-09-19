"""DVID-native ground-truth region materialization and split safety checks."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from .io import sha256_file, write_json_atomic


RAW_PREFIX = "https://waspem-dvid2.flatironinstitute.org/api/node/aa49d16e88424cdbae89f4b8ced2a49b/five_yuri_4contrast/raw/0_1_2/"

# Immutable target-domain regression fixture. It cannot enter annotation,
# training, or model selection for target adaptation.
REGRESSION_ONLY_REGION_IDS = {"MV-GTVOL-000004"}


def _load_plan(path: Path) -> dict:
    """Read a declared DVID-native label plan as an immutable input contract."""
    # The manifest is JSON, which is a valid YAML subset, to retain a
    # dependency-free parser for an evidence-critical boundary.
    return json.loads(path.read_text(encoding="utf-8"))


def _separation(a: dict, b: dict) -> int:
    """Chebyshev gap between half-open XYZ boxes; zero means overlap/touch."""
    gaps = []
    for axis in ("x", "y", "z"):
        first, second = a[axis], b[axis]
        gaps.append(max(0, second[0] - first[1], first[0] - second[1]))
    return max(gaps)


def validate_plan(plan: dict) -> None:
    """Enforce spatial isolation, split assignment, and protected regression use."""
    regions = plan["regions"]
    required = int(plan["required_excluded_buffer_voxels"])
    identifiers = [item["id"] for item in regions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Ground-truth region identifiers must be unique")
    for index, region in enumerate(regions):
        bounds = region["bounds_xyz"]
        if region["split"] not in {"train", "validation", "test", "regression"}:
            raise ValueError(f"{region['id']}: invalid split")
        if any(bounds[axis][0] >= bounds[axis][1] for axis in ("x", "y", "z")):
            raise ValueError(f"{region['id']}: invalid bounds")
        for other in regions[index + 1:]:
            if region["split"] != other["split"] and _separation(bounds, other["bounds_xyz"]) < required:
                raise ValueError(f"{region['id']} and {other['id']} violate {required}-voxel cross-split buffer")


def materialize_regions(plan_path: Path, cache: Path, manifest: Path) -> dict:
    """Fetch planned raw cubes without presenting cached EM as human labels."""
    plan = _load_plan(plan_path)
    validate_plan(plan)
    source = plan["source"]
    records = []
    for region in plan["regions"]:
        bounds = region["bounds_xyz"]
        dimensions = [bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z")]
        url = RAW_PREFIX + f"{dimensions[0]}_{dimensions[1]}_{dimensions[2]}/{bounds['x'][0]}_{bounds['y'][0]}_{bounds['z'][0]}"
        destination = cache / region["id"]
        destination.mkdir(parents=True, exist_ok=True)
        raw_path = destination / "raw_zyx.npy"
        if not raw_path.exists():
            with urlopen(url, timeout=180) as response:
                payload = response.read()
            expected = dimensions[0] * dimensions[1] * dimensions[2]
            if len(payload) != expected:
                raise ValueError(f"{region['id']}: DVID response {len(payload)} bytes != expected {expected}")
            np.save(raw_path, np.frombuffer(payload, dtype=np.uint8).reshape((dimensions[2], dimensions[1], dimensions[0])), allow_pickle=False)
        records.append({
            **region,
            "volume_id": source["volume_id"], "dataset_id": source["dataset_id"], "coordinate_frame": source["coordinate_frame"],
            "physical_bounds_nm_xyz": {axis: [value * 8.0 for value in bounds[axis]] for axis in ("x", "y", "z")},
            "dimensions_voxels_xyz": dimensions, "source_url": url, "raw_path": str(raw_path.resolve()), "raw_sha256": sha256_file(raw_path),
            "label_path": None, "label_sha256": None, "instance_count": 0, "labeled_voxels": 0,
            "review_status": "UNANNOTATED", "ambiguities": 0, "raw_data_immutable": True,
        })
    value = {"schema_version": "1.0", "id": "MV-DVID-GT-v1-CANDIDATE", "status": "RAW_REGIONS_MATERIALIZED_AWAITING_QUALIFIED_LABELS", "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "plan_sha256": sha256_file(plan_path), "regions": records}
    write_json_atomic(manifest, value)
    return {"regions": len(records), "raw_regions": len(records), "labeled_regions": 0, "status": value["status"]}


def register_instance_label(manifest_path: Path, region_id: str, label_path: Path, reviewer: str, review_status: str, objects_path: Path) -> dict:
    """Register a reviewer-supplied DVID-native instance mask without altering it.

    Annotation UIs are deliberately external to this importer.  The resulting
    mask must match the cached raw volume exactly and only carries claims
    warranted by its declared review state.
    """
    allowed = {"DRAFT", "FIRST_PASS", "REVIEW_REQUIRED", "REVIEWED", "SECOND_PASS_REVIEWED", "GOLD_STANDARD", "AMBIGUOUS"}
    if review_status not in allowed or not reviewer.strip():
        raise ValueError("A supported review state and non-empty reviewer are required")
    if region_id in REGRESSION_ONLY_REGION_IDS:
        raise ValueError(f"{region_id} is immutable regression-only and cannot receive a target-adaptation label")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    region = next((item for item in manifest["regions"] if item["id"] == region_id), None)
    if region is None:
        raise ValueError(f"Unknown ground-truth region {region_id}")
    labels = np.load(label_path, allow_pickle=False)
    expected_zyx = tuple(reversed(region["dimensions_voxels_xyz"]))
    if labels.shape != expected_zyx or labels.ndim != 3:
        raise ValueError(f"{region_id}: label shape {labels.shape} does not match raw shape {expected_zyx}")
    if labels.dtype.kind not in {"u", "i"} or (labels < 0).any():
        raise ValueError("Instance labels must be a non-negative integer 3-D array")
    instances, counts = np.unique(labels, return_counts=True)
    positive = [(int(identifier), int(count)) for identifier, count in zip(instances, counts, strict=True) if identifier > 0]
    if not positive:
        raise ValueError("A ground-truth instance label must contain at least one positive identity")
    destination = objects_path.parent / "labels" / f"{region_id}-labels_zyx.npy"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"Refusing to overwrite registered label artifact {destination}")
    # Preserve an immutable, byte-identical copy under the ground-truth store.
    destination.write_bytes(label_path.read_bytes())
    region.update({"label_path": str(destination.resolve()), "label_sha256": sha256_file(destination), "instance_count": len(positive), "labeled_voxels": int((labels > 0).sum()), "review_status": review_status, "reviewer": reviewer})
    objects = json.loads(objects_path.read_text(encoding="utf-8")) if objects_path.exists() else {"objects": []}
    if any(item["region_id"] == region_id for item in objects["objects"]):
        raise ValueError(f"Ground-truth objects already registered for {region_id}")
    for ordinal, (source_label, voxel_count) in enumerate(positive, 1):
        objects["objects"].append({"id": f"MV-GT-N-{region_id[-6:]}-{ordinal:05d}", "region_id": region_id, "source_instance_label": source_label, "voxel_count": voxel_count, "review_status": review_status, "reviewer": reviewer, "boundary_status": "GT_VOLUME_TRUNCATED_OR_UNREVIEWED", "ambiguity": review_status == "AMBIGUOUS"})
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(objects_path, objects)
    return {"region_id": region_id, "instances": len(positive), "labeled_voxels": region["labeled_voxels"], "label_sha256": region["label_sha256"], "review_status": review_status}
