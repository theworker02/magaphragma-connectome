"""Freeze the spatially diverse G2 DVID supervision cohort before inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mvconnectome.spatial_expansion import create_g2_review_inputs, freeze_spatial_expansion_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-manifest", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, help="New frozen cohort path")
    parser.add_argument("--existing-cohort", type=Path, help="Existing frozen cohort; only creates review inputs")
    parser.add_argument("--review-inputs", type=Path, help="Create raw-only external review request/manifest after freezing")
    args = parser.parse_args()
    if args.existing_cohort:
        if args.survey_manifest or args.plan or args.output:
            parser.error("--existing-cohort cannot be combined with cohort-freezing arguments")
        cohort_path = args.existing_cohort
        result = json.loads(cohort_path.read_text(encoding="utf-8"))
    else:
        if not args.survey_manifest or not args.plan or not args.output:
            parser.error("--survey-manifest, --plan, and --output are required when freezing a cohort")
        result = freeze_spatial_expansion_cohort(
            survey_manifest_path=args.survey_manifest, plan_path=args.plan, output_path=args.output,
        )
        cohort_path = args.output
    output = {"id": result["id"], "status": result["status"], "regions": len(result["regions"])}
    if args.review_inputs:
        output["review_inputs"] = create_g2_review_inputs(cohort_path=cohort_path, output_dir=args.review_inputs)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
