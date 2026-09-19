"""DeltaGraph ? connectome structural diff."""
from __future__ import annotations

from .__version__ import __version__
from .causes import infer
from .diff import Build, DeltaGraph, DeltaResult

__all__ = ["__version__", "Build", "DeltaGraph", "DeltaResult", "infer"]
