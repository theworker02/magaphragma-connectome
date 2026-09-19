"""Fail-closed target-domain adaptation gates for real DVID labels."""
from __future__ import annotations

import json
from pathlib import Path

from .io import sha256_file


ALLOWED_REVIEW_STATES = {"REVIEWED", "SECOND_PASS_REVIEWED", "GOLD_STANDARD"}


def adaptation_status(plan_path: Path, manifest_path: Path) -> dict:
    """Report whether independently reviewed labels unlock DVID fine tuning."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = plan["roles"]
    regression_id = roles["regression_only"]
    regions = {region["id"]: region for region in manifest["regions"]}
    if regression_id in set(roles["train"]) | set(roles["validation"]):
        raise ValueError("Regression-only region appears in target adaptation split")
    if regression_id not in regions:
        raise ValueError("Regression-only region absent from source manifest")
    if regions[regression_id].get("label_path"):
        raise ValueError("Regression-only region has a label artifact")

    def qualified(ids: list[str]) -> list[str]:
        return [region_id for region_id in ids if (region := regions.get(region_id))
                and region.get("label_path") and region.get("review_status") in ALLOWED_REVIEW_STATES]

    train = qualified(roles["train"])
    validation = qualified(roles["validation"])
    missing = [region_id for region_id in roles["train"] + roles["validation"] if region_id not in train + validation]
    ready = bool(train and validation)
    return {
        "id": plan["id"],
        "status": "READY_FOR_DVID_ADAPTATION" if ready else "BLOCKED_AWAITING_REVIEWED_DVID_NATIVE_LABELS",
        "plan_sha256": sha256_file(plan_path),
        "regression_only": regression_id,
        "qualified_train_regions": train,
        "qualified_validation_regions": validation,
        "unqualified_or_missing_review_regions": missing,
        "training_unlock_rule": "at_least_one_reviewed_train_region_and_one_distinct_reviewed_validation_region",
        "prohibited": ["machine_pseudolabel_as_ground_truth", "regression_cube_labeling", "cross_split_training"],
    }
