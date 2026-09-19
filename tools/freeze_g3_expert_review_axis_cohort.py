"""Freeze an axis-balanced expert-review cohort from frozen G3 supervision.

Exits non-zero when no affinity axis satisfies the TRAIN+VALIDATION
SAME+DIFFERENT contract; the auditable failure report is still written to
--output so the next required reviewed supervision is on the record.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mvconnectome.g3_expert_review_axis_cohort import (
    AxisCohortFailure,
    freeze_g3_expert_review_axis_cohort,
)


parser = argparse.ArgumentParser()
parser.add_argument("--supervision-manifest", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--cohort-id", default=None)
args = parser.parse_args()

try:
    receipt = freeze_g3_expert_review_axis_cohort(
        supervision_manifest_path=args.supervision_manifest,
        output_path=args.output,
        cohort_id=args.cohort_id,
    )
except AxisCohortFailure as failure:
    print(json.dumps(failure.report, indent=2, sort_keys=True))
    print(f"\nAXIS COHORT FAILED: {failure}", file=sys.stderr)
    raise SystemExit(1)

print(json.dumps(receipt, indent=2, sort_keys=True))
