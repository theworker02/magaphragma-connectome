"""RegionBench-lite: frozen synthetic benchmark regions for comparable metrics."""
from .bench import RegionBench, ensure_frozen, list_regions, run_region
__all__ = ["RegionBench", "ensure_frozen", "list_regions", "run_region"]
