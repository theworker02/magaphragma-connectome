"""Create a frozen G1-assisted expert-review queue for one frozen G2 region."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mvconnectome.g2_diagnostics import create_g2_diagnostic_queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()
    queue = create_g2_diagnostic_queue(cohort_path=args.cohort, region_id=args.region, inference_dir=args.inference,
        output_path=args.output, expected_checkpoint_sha256=args.checkpoint_sha256)
    print(json.dumps({"id": queue["id"], "questions": len(queue["questions"]), "status": queue["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
