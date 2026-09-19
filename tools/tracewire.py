#!/usr/bin/env python3
"""Thin entry ? TraceWire."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.tracewire.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
