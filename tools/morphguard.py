#!/usr/bin/env python3
"""Connectome tools entry ? MorphGuard."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for p in (REPO, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from morphguard.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
