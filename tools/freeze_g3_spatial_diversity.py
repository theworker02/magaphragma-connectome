"""Freeze the G3 spatial-diversity cohort before interface review."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))
from mvconnectome.g3_spatial_diversity import freeze_g3_cohort
p=argparse.ArgumentParser(); p.add_argument("--survey",type=Path); p.add_argument("--plan",type=Path); p.add_argument("--output",type=Path); p.add_argument("--existing-cohort",type=Path); p.add_argument("--review-inputs",type=Path); a=p.parse_args()
if a.existing_cohort:
    if a.survey or a.plan or a.output or not a.review_inputs: p.error("--existing-cohort requires only --review-inputs")
    from mvconnectome.g3_spatial_diversity import create_g3_review_inputs
    print(json.dumps(create_g3_review_inputs(cohort_path=a.existing_cohort,output_dir=a.review_inputs),indent=2))
else:
    if not a.survey or not a.plan or not a.output: p.error("--survey, --plan, --output required")
    print(json.dumps(freeze_g3_cohort(survey_path=a.survey,plan_path=a.plan,output_path=a.output),indent=2))
