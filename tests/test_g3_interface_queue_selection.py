"""Selection/geometry tests for the G3 interface-queue generator.

The raw volumes here are synthetic geometry fixtures used only to exercise
candidate *selection* (ranking, exclusion, orientation, paging). They are not
biological evidence and never enter any SAME/DIFFERENT gate.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "tools" / "generate_g3_interface_queue.py"


def _write_workspace(root: Path, *, crop_id="MV-G3-TRAIN-TEST", split="G3_TARGET_TRAIN", seed=0) -> Path:
    import hashlib

    rng = np.random.default_rng(seed)
    raw = (rng.random((24, 40, 40)) * 255).astype(np.uint8)
    raw_path = root / f"{crop_id}-raw.npy"
    np.save(raw_path, raw, allow_pickle=False)
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    workspace = {
        "id": f"MV-EXTERNAL-BOUNDARY-WORKSPACE-{crop_id}",
        "crop_id": crop_id,
        "split": split,
        "status": "EXTERNAL_EXPERT_RAW_EM_REVIEW_REQUIRED",
        "raw": {"path": str(raw_path.resolve()), "sha256": raw_sha, "shape_zyx": list(raw.shape), "dtype": "uint8"},
        "event_log": {"append_only": True, "path": str((root / f"{crop_id}.events.jsonl").resolve())},
    }
    workspace_path = root / f"{crop_id}-workspace.json"
    workspace_path.write_text(json.dumps(workspace))
    return workspace_path


def _run(workspace: Path, output: Path, *args: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(GENERATOR), str(workspace), "--output", str(output), *args],
        capture_output=True, text=True, cwd=str(REPO),
    )
    if result.returncode != 0:
        raise RuntimeError(f"generator failed: {result.stderr}\n{result.stdout}")
    return json.loads(result.stdout)


def _pairs(queue_path: Path) -> set:
    queue = json.loads(queue_path.read_text())
    return {(tuple(q["pair_left_zyx"]), tuple(q["pair_right_zyx"]), int(q["channel_zyx"])) for q in queue["questions"]}


def test_default_behavior_is_deterministic_and_unchanged(tmp_path: Path) -> None:
    ws = _write_workspace(tmp_path, seed=1)
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    _run(ws, out_a, "--interfaces", "6")
    _run(ws, out_b, "--interfaces", "6")
    a = json.loads(out_a.read_text())
    b = json.loads(out_b.read_text())
    # Volatile field only is created_at; strip and compare the rest.
    a.pop("created_at")
    b.pop("created_at")
    assert a == b
    # Default selection records the extension fields as inert defaults.
    assert a["selection"]["skip_candidates"] == 0
    assert a["selection"]["orientation_axis"] is None
    assert a["selection"]["excluded_pair_count"] == 0


def test_orientation_axis_restricts_to_that_channel(tmp_path: Path) -> None:
    ws = _write_workspace(tmp_path, seed=2)
    out = tmp_path / "y.json"
    _run(ws, out, "--interfaces", "5", "--orientation-axis", "Y")
    queue = json.loads(out.read_text())
    assert queue["questions"]
    assert all(q["channel_zyx"] == 1 for q in queue["questions"])
    assert queue["selection"]["orientation_axis"] == "Y"
    assert queue["selection"]["orientation_axis_channel_zyx"] == 1


def test_exclude_queue_removes_prior_exact_pairs(tmp_path: Path) -> None:
    ws = _write_workspace(tmp_path, seed=3)
    first = tmp_path / "first.json"
    _run(ws, first, "--interfaces", "5", "--orientation-axis", "Y")
    second = tmp_path / "second.json"
    _run(ws, second, "--interfaces", "5", "--orientation-axis", "Y", "--exclude-queue", str(first))
    first_pairs = _pairs(first)
    second_pairs = _pairs(second)
    # No exact pair from the excluded queue may reappear.
    assert first_pairs.isdisjoint(second_pairs)
    queue = json.loads(second.read_text())
    assert queue["selection"]["exclusion_sources"][0]["excluded_questions"] == len(json.loads(first.read_text())["questions"])


def test_paging_is_deterministic_and_advances(tmp_path: Path) -> None:
    ws = _write_workspace(tmp_path, seed=4)
    page0 = tmp_path / "p0.json"
    page0b = tmp_path / "p0b.json"
    page1 = tmp_path / "p1.json"
    _run(ws, page0, "--interfaces", "3", "--orientation-axis", "Y")
    _run(ws, page0b, "--interfaces", "3", "--orientation-axis", "Y")
    _run(ws, page1, "--interfaces", "3", "--orientation-axis", "Y", "--skip-candidates", "3")
    # Same params -> identical selection (deterministic).
    assert _pairs(page0) == _pairs(page0b)
    # Paging past the first tier yields a different top interface centre.
    q0 = json.loads(page0.read_text())["questions"][0]["pair_left_zyx"]
    q1 = json.loads(page1.read_text())["questions"][0]["pair_left_zyx"]
    assert q0 != q1


def test_selection_never_reads_labels(tmp_path: Path) -> None:
    # The generator is given only a workspace + raw. There is no channel by
    # which a SAME/DIFFERENT label could enter selection; assert the recorded
    # contract makes that explicit and that prohibited_inputs lists labels.
    ws = _write_workspace(tmp_path, seed=5)
    out = tmp_path / "blind.json"
    _run(ws, out, "--interfaces", "4", "--orientation-axis", "Y")
    queue = json.loads(out.read_text())
    assert queue["selection"]["label_blind"] is True
    assert "reviewed SAME/DIFFERENT decisions" in queue["selection"]["prohibited_inputs"]


def test_refuses_to_overwrite_existing_queue(tmp_path: Path) -> None:
    ws = _write_workspace(tmp_path, seed=6)
    out = tmp_path / "once.json"
    _run(ws, out, "--interfaces", "3")
    result = subprocess.run(
        [sys.executable, str(GENERATOR), str(ws), "--output", str(out), "--interfaces", "3"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr
