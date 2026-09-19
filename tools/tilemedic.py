#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from tilemedic.cli import main
raise SystemExit(main())
