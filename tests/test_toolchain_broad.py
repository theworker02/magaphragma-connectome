"""Broadened toolchain tests — UTF-8 only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]


def test_artifactvet_npy_and_json(tmp_path: Path):
    from _core.artifactvet import ArtifactVet

    arr = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    npy = tmp_path / "t.npy"
    np.save(npy, arr, allow_pickle=False)
    r = ArtifactVet().vet_npy(npy, expected_shape=(2, 3, 4), expected_dtype="uint8")
    assert r.ok
    assert r.meta["shape"] == [2, 3, 4]

    man = tmp_path / "m.json"
    man.write_text(json.dumps({"schema": "x", "items": []}) + chr(10), encoding="utf-8")
    j = ArtifactVet().vet_json(man, required_keys=("schema", "items"))
    assert j.ok


def test_regionbench_frozen_and_smoke():
    from regionbench.bench import ensure_frozen, RegionBench, REGIONS
    from voxscout.scout import VoxScout

    specs = ensure_frozen()
    assert len(specs) == len(REGIONS)
    assert (REPO / "receipts" / "regionbench" / "index.json").exists()

    def metrics(rid, vol):
        tile = (16, 16, 16) if min(vol.shape) >= 32 else tuple(max(8, s // 2) for s in vol.shape)
        pm = VoxScout(tile).analyze(vol)
        return {"n_tiles": len(pm.tiles)}

    results = RegionBench().run("voxscout", metrics)
    assert all(r.ok for r in results)
    assert len(results) == len(REGIONS)


def test_pipeline_plan_and_close_gaps():
    from pipeline.flows import plan_pipeline, close_gaps_pipeline

    plan = plan_pipeline(shape_zyx=(32, 32, 32), tile_zyx=(16, 16, 16))
    assert plan["pipeline"] == "plan"
    assert plan["steps"][0]["n_tiles"] == 8
    assert Path(plan["hints"]).exists()
    assert plan["artifactvet"]["ok"]

    close = close_gaps_pipeline(shape_zyx=(64, 64, 64), run_tilemedic=True)
    assert close["pipeline"] == "close-gaps"
    assert "repair_plan" in close
    assert Path(close["receipt"]).exists()


def test_edgeprobe_bundle():
    from edgeprobe.probe import edgeprobe

    summary = edgeprobe("nA", "nB")
    assert summary["pre"] == "nA"
    assert summary["post"] == "nB"
    bundle = Path(summary["bundle"])
    assert (bundle / "summary.json").exists()
    assert (bundle / "back.txt").exists()


def test_neurocache_axonforge_identity_roundtrip():
    from neurocache.axonforge_id import demo_roundtrip

    result = demo_roundtrip("T-Z001-Y002-X003")
    assert result["lookup_hit"] is True
    assert result["payload"]["tile_key"] == "T-Z001-Y002-X003"
    assert len(result["identity_hash"]) == 64


def test_runddoctor_imports_and_switch():
    from runddoctor.doctor import RunDoctor

    report = RunDoctor().run()
    assert report.switch
    assert "powered_on" in report.switch
    assert report.tool_imports
    assert any("pipeline" in r or "tools" in r or "gaphound" in r or "tilemedic" in r for r in report.recommended)
    text = report.format_summary()
    assert "AxonForge switch" in text or "powered_on" in text


def test_seamsmith_branchjudge_on_review():
    from seamsmith.branch_assist import assist_review_candidates

    cands = [
        {
            "recommended_action": "REVIEW",
            "a": {"label_id": 1},
            "b": {"label_id": 2},
            "evidence": {"spatial_distance": 4.0, "confidence": 0.4},
        },
        {"recommended_action": "MATCH", "a": {"label_id": 3}, "b": {"label_id": 3}, "evidence": {}},
    ]
    out = assist_review_candidates(cands)
    assert len(out) == 1
    assert "branchjudge" in out[0]
    assert cands[0]["branchjudge"]["decision"] in ("REVIEW", "MERGE", "SPLIT")
