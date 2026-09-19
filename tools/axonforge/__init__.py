"""AxonForge tool package — re-exports top-level axonforge + switch."""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from axonforge.switch import SWITCH, LightSwitch
__all__ = ["SWITCH", "LightSwitch"]
