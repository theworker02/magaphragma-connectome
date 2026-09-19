from axonforge.graph import SpatialWorkGraph
from axonforge.ledger import CompletenessLedger
from axonforge.gates import equivalence_gate, speed_gate
from axonforge.backends import HaloCache, reference_infer
from axonforge.adaptive import run_adaptive
from axonforge.runtime import Runtime
from axonforge.__version__ import __version__
import numpy as np


def test_version():
    assert __version__ == "1.4.0"


def test_work_graph_covers_volume():
    g = SpatialWorkGraph.build((64, 128, 128), (32, 64, 64), load_existing=False)
    assert len(g.tasks) == 2 * 2 * 2
    assert sum(t.voxels for t in g.tasks.values()) == 64 * 128 * 128


def test_equivalence_gate():
    assert equivalence_gate("abc", "abc").passed
    assert not equivalence_gate("abc", "xyz").passed


def test_halo_cache_hit():
    cache = HaloCache()
    tile = np.full((16, 32, 32), 40, dtype=np.uint8)
    a = cache.infer(tile)
    b = cache.infer(tile)
    assert a.checksum == b.checksum
    assert b.cache_hit
    assert cache.hit_rate > 0


def test_adaptive_completeness():
    g = SpatialWorkGraph.build((64, 64, 64), (32, 32, 32), load_existing=False)
    led = CompletenessLedger(total_voxels=64 * 64 * 64)
    report = run_adaptive(g, led, use_cascade=False)
    assert report.ledger["validated_coverage"] == 1.0
    assert report.ledger["unaccounted_volume"] == 0
    assert report.claims.get("halo_cache") in (True, False)


def test_runtime_demo():
    rt = Runtime()
    sub = rt.submit_volume(shape_zyx=(64, 64, 64), tile_zyx=(32, 32, 32), reset=True)
    assert sub["ok"]
    out = rt.request_inference(run_adaptive=True)
    assert out["ok"]
    st = rt.status()
    assert st["version"] == "1.4.0"
    assert st["telemetry"]["validated"] > 0


def test_speed_gate():
    assert speed_gate(1.0, 2.0).passed
    assert not speed_gate(2.0, 2.0).passed
