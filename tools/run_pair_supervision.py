"""Run the dedicated DVID pair-supervision module without importing the full CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

from mvconnectome.pair_supervision import build_pair_supervision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("crop_id")
    parser.add_argument("--affinity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("ground_truth/annotation_crops_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    print(build_pair_supervision(parser.parse_args().manifest, parser.parse_args().crop_id, parser.parse_args().affinity, parser.parse_args().output))


if __name__ == "__main__":
    main()
