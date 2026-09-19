"""Freeze the reviewed G2 DVID affinity dataset after all expert decisions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mvconnectome.g2_supervision import freeze_g2_supervision


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cohort", type=Path, required=True)
parser.add_argument("--receipts", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
print(json.dumps(freeze_g2_supervision(cohort_path=args.cohort, receipt_root=args.receipts, output_path=args.output), indent=2))
