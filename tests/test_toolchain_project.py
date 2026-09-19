"""Project-level wiring: every registered tool connects into Connectome."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "src")]


def test_connection_status_all_wired():
    from mvconnectome.toolchain import connection_status

    status = connection_status()
    assert status["ok"] is True
    assert status["n_tools"] == 17
    assert status["n_importable"] == 17
    assert status["unwired"] == []
    assert status["missing_imports"] == []


def test_post_run_toolchain_invokes_all():
    from axonforge.bridge import post_run_toolchain

    report = post_run_toolchain((32, 32, 32), tiles_done=4)
    expected = {
        "gaphound", "runddoctor", "morphguard", "seamsmith", "branchjudge",
        "synapselens", "deltagraph", "edgeprobe", "hyperdrain", "artifactvet",
    }
    invoked = set(report.get("tools_invoked") or [])
    assert expected <= invoked
    for key in expected:
        assert "error" not in (report.get(key) or {}), (key, report.get(key))


def test_expanded_pipelines():
    from pipeline.flows import (
        reconcile_pipeline, audit_pipeline, connectivity_pipeline,
        diff_pipeline, mass_status_pipeline,
    )

    assert reconcile_pipeline()["pipeline"] == "reconcile"
    assert audit_pipeline()["pipeline"] == "audit"
    assert connectivity_pipeline()["pipeline"] == "connectivity"
    assert diff_pipeline()["pipeline"] == "diff"
    assert mass_status_pipeline()["pipeline"] == "mass-status"


def test_runtime_report_includes_toolchain():
    from axonforge.runtime import Runtime
    from axonforge.switch import SWITCH

    SWITCH.set_power(True)
    rt = Runtime()
    sub = rt.submit_volume(shape_zyx=(32, 32, 32), tile_zyx=(16, 16, 16), reset=True)
    assert sub["ok"]
    assert "voxscout_hints" in sub
    out = rt.request_inference()
    assert out["ok"]
    report = out["report"]
    assert "toolchain_report" in report
    assert "gaphound" in report
    tools = set(report["toolchain"])
    assert "morphguard" in tools or "gaphound" in tools
    st = rt.status()
    assert len(st["toolchain"]["connected"]) >= 17


def test_vigilia_toolchain_forward():
    from mvconnectome.cli import main

    assert main(["toolchain", "status"]) == 0

