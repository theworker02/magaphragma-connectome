"""Materialize reviewed interface-level G3 affinity supervision."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mvconnectome.g3_supervision import freeze_g3_supervision


parser = argparse.ArgumentParser()
parser.add_argument("--cohort", required=True, type=Path)
parser.add_argument("--bindings", required=True, type=Path)
parser.add_argument("--output-root", required=True, type=Path)
parser.add_argument("--manifest", required=True, type=Path)
args = parser.parse_args()
print(json.dumps(freeze_g3_supervision(cohort_path=args.cohort, inputs_path=args.bindings, output_root=args.output_root, manifest_path=args.manifest), indent=2))
