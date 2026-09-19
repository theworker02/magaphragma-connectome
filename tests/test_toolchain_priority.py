import numpy as np
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]


def test_voxscout_covers_all_tiles():
    from voxscout.scout import VoxScout
    vol = np.zeros((64, 64, 64), dtype=np.uint8)
    vol[20:40, 20:40, 20:40] = 128
    pm = VoxScout((32, 32, 32)).analyze(vol)
    assert len(pm.tiles) == 8
    assert sum(pm.class_counts().values()) == 8
    hints = pm.as_axonforge_hints()
    assert all("optimize_ok" in h for h in hints.values())


def test_gaphound_repair_plan_minimum():
    from gaphound.scan import GapHound
    report = GapHound().scan((100, 100, 100))
    plan = GapHound().repair_plan(report)
    assert "worklist" in plan
    assert plan["n_work_items"] == len(plan["worklist"])


def test_tilemedic_oom_safe():
    from tilemedic.medic import TileMedic, PROTECTED_KEYS
    r = TileMedic().recover(
        {"tile_key": "t1", "error": "HIP out of memory"},
        config={"batch_size": 16, "checkpoint": "x"},
    )
    assert r.diagnosis == "OOM"
    assert r.safe
    assert "batch_size" in r.altered_keys
    assert "checkpoint" not in r.altered_keys
    assert not (set(r.altered_keys) & PROTECTED_KEYS)


def test_branchjudge_evidence_note():
    from branchjudge.judge import BranchJudge
    a = np.array([[0, 0, 0], [0, 0, 5]], float)
    b = np.array([[0, 0, 8.8], [0, 0, 12]], float)
    v = BranchJudge().judge(a, b, diameter_a=2, diameter_b=2, seg_confidence=0.4)
    assert v.decision in ("REVIEW", "MERGE", "SPLIT")
    assert "evidence" in v.as_dict()["note"].lower()


def test_switch_toggle():
    from axonforge.switch import LightSwitch
    sw = LightSwitch()
    before = sw.is_on()
    sw.toggle()
    assert sw.is_on() != before
    sw.set_power(True)
